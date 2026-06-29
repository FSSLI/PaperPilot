"""
RAG 评测脚本 - 基于 Ragas 框架

使用 Ragas 框架对 PaperPilot 的 RAG 管线进行系统评测，
覆盖 Faithfulness / Answer Relevancy / Context Precision / Context Recall 四项指标，
以及引用溯源相关的 Citation Accuracy 指标。

运行方式：
    1. 确保 mildoc_wxkf 服务已配置（.env 文件中 LLM、Embedding、Milvus 等配置正确）
    2. 将待评测的文献上传到 MinIO 知识库（确保 mildoc_index 已完成索引）
    3. 在 eval/ 目录下执行：
       python eval_rag.py
    4. 可选参数：
       --dataset eval_dataset.json    评测数据集路径（默认同级目录下的 eval_dataset.json）
       --output eval_results.json        结果输出路径（默认 eval_results.json）
       --no-rerank                       关闭 rerank 进行评测
       --limit N                         仅评测前 N 条 QA

依赖安装：
    pip install ragas datasets langchain-openai langchain-community
"""

import os
import re
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 配置日志
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_rag")

# ─────────────────────────────────────────────────────────────
# 兼容处理：ragas 0.4.x 强制导入 langchain_community.chat_models.vertexai，
# 但 langchain-community >= 0.3 已移除该模块。
# 这里动态注入一个 stub 模块，避免 ragas 加载失败。
# ─────────────────────────────────────────────────────────────
import types
import importlib

def _patch_vertexai_stub():
    """为 ragas 注入 langchain_community.chat_models.vertexai 的 stub 模块"""
    try:
        import langchain_community.chat_models
        if not hasattr(langchain_community.chat_models, 'vertexai'):
            stub = types.ModuleType('langchain_community.chat_models.vertexai')
            stub.ChatVertexAI = type('ChatVertexAI', (), {})
            sys.modules['langchain_community.chat_models.vertexai'] = stub
            langchain_community.chat_models.vertexai = stub
    except ImportError:
        pass

_patch_vertexai_stub()

# ─────────────────────────────────────────────────────────────
# 延迟导入 Ragas（首次运行前会自动检查）
# ─────────────────────────────────────────────────────────────
_RAGAS_IMPORT_ERROR = ""
try:
    from datasets import Dataset as HFDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    HAS_RAGAS = True
except ImportError as e:
    HAS_RAGAS = False
    _RAGAS_IMPORT_ERROR = str(e)


# ═══════════════════════════════════════════════════════════════
# 引用溯源辅助函数
# ═══════════════════════════════════════════════════════════════

def extract_citations(answer: str) -> List[str]:
    """从回答文本中提取所有引用的文件名。

    匹配格式：【来源：《filename》】
    返回去重后的文件名列表。
    """
    pattern = r"【来源：《([^》]+)》】"
    matches = re.findall(pattern, answer)
    # 去重保持顺序
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def compute_citation_metrics(
    answer: str,
    cited_files: List[str],
    retrieved_files: List[str],
    ground_truth_file: str,
) -> Dict[str, Any]:
    """计算单条 QA 的引用相关指标。

    - citation_recall: 引用的文档是否覆盖了正确答案来源（召回率视角）
    - citation_precision: 引用的文档中有多少是真正被检索到的（精确率视角）
    - citation_accuracy: 引用的文档是否包含正确答案来源（最严格指标）
    - has_citation: 是否有任何引用
    """
    has_citation = len(cited_files) > 0

    # Citation Recall：引用中是否包含正确答案来源
    citation_recall = 1.0 if ground_truth_file in cited_files else 0.0

    # Citation Precision：引用的文档中，有多少在检索结果中
    if cited_files:
        cited_in_retrieved = sum(1 for f in cited_files if f in retrieved_files)
        citation_precision = cited_in_retrieved / len(cited_files)
    else:
        citation_precision = 0.0

    # Citation Accuracy（最严格）：是否引用了正确答案来源
    citation_accuracy = 1.0 if ground_truth_file in cited_files else 0.0

    return {
        "has_citation": has_citation,
        "cited_files": cited_files,
        "citation_recall": citation_recall,
        "citation_precision": round(citation_precision, 4),
        "citation_accuracy": citation_accuracy,
    }


# ═══════════════════════════════════════════════════════════════
# 核心评测逻辑
# ═══════════════════════════════════════════════════════════════

class RAGEvaluator:
    """RAG 评测器

    接入 PaperPilot 的 RAG 管线，逐条采集检索上下文和 LLM 回答，
    然后使用 Ragas 框架计算评测指标。
    """

    def __init__(self, use_rerank: bool = True):
        self.use_rerank = use_rerank

        # 延迟导入项目内部模块（config 在 wxkf/ 目录，需要将 wxkf/ 加入路径）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wxkf_dir = os.path.join(project_root, "mildoc_wxkf")
        sys.path.insert(0, wxkf_dir)
        from config import Config
        from langchain_openai import ChatOpenAI
        from langchain_community.callbacks.manager import get_openai_callback

        self.Config = Config
        self.get_openai_callback = get_openai_callback

        # 初始化 Embedding 模型（用于向量检索）
        logger.info("正在初始化 Embedding 模型...")
        from openai import OpenAI

        class CustomEmbeddings:
            def __init__(self, model_name, api_key, api_base, dimensions):
                self.client = OpenAI(api_key=api_key, base_url=api_base)
                self.model_name = model_name
                self.dimensions = dimensions

            def embed_query(self, text):
                return self.embed_documents([text])[0]

            def embed_documents(self, texts):
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
                return [d.embedding for d in response.data]

        self.embeddings = CustomEmbeddings(
            model_name=Config.LLM_EMBEDDING_MODEL_NAME,
            api_key=Config.LLM_EMBEDDING_API_KEY,
            api_base=Config.LLM_EMBEDDING_BASE_URL,
            dimensions=Config.MILVUS_VECTOR_DIM,
        )

        # 初始化 Milvus 检索（使用 MilvusClient，绕过 langchain_milvus 的 bug）
        logger.info("正在初始化 Milvus 检索...")
        from pymilvus import MilvusClient
        uri = f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}"
        connect_kwargs = {"uri": uri, "db_name": Config.MILVUS_DATABASE}
        if Config.MILVUS_USER:
            connect_kwargs["user"] = Config.MILVUS_USER
        if Config.MILVUS_PASSWORD:
            connect_kwargs["password"] = Config.MILVUS_PASSWORD
        self._milvus_client = MilvusClient(**connect_kwargs)
        logger.info(f"Milvus 检索初始化成功: {uri}, db={Config.MILVUS_DATABASE}")

        # 初始化 LLM
        logger.info("正在初始化 LLM...")
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL_NAME,
            openai_api_key=Config.LLM_API_KEY,
            openai_api_base=Config.LLM_BASE_URL,
            temperature=0.1,
        )

        # 初始化重排序服务
        from rerank_service import get_rerank_service
        self.rerank_service = get_rerank_service()
        if self.rerank_service:
            logger.info("重排序服务初始化成功")
        else:
            logger.info("重排序服务未配置，跳过 rerank")

        # 初始化 Ragas Judge LLM
        self.judge_llm = ChatOpenAI(
            model=Config.LLM_MODEL_NAME,
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            temperature=0.1,
        )

        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        self.ragas_llm = LangchainLLMWrapper(self.judge_llm)
        self.judge_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

        # 提示词模板（与 RAGService 保持一致，含引用溯源）
        self.PROMPT_TEMPLATE = """你是一位专业的客服人员，请根据提供的知识库内容来回答用户的问题。

知识库内容:
{context}

用户问题: {question}

回答要求：
1. 【角色定位】你是一位专业、耐心、友善的客服代表
2. 【回答原则】严格基于知识库内容回答，不得编造或推测信息
3. 【引用溯源】每当你引用知识库中的具体信息时，必须在引用处标注来源，格式为【来源：《文件名》】
   - 例如：「该方法在ImageNet上达到了95.3%的准确率【来源：《ResNet.pdf》】」
   - 如果回答涉及多个文献，用多个【来源：《文件名》】分别标注
   - 综合性结论应引用多个相关文献
4. 【准确性要求】
   - 如果知识库中有明确答案，请准确完整地回答
   - 如果知识库中信息不完整，说明现有信息并提示用户可联系人工客服获取更详细信息
   - 如果知识库中完全没有相关信息，请礼貌地说明无法找到相关资料，建议用户转接人工客服
5. 【回答格式】
   - 使用纯文本格式，不使用markdown格式
   - 语言简洁明了，适合微信对话环境
   - 使用礼貌、专业的语调
   - 如需列举，使用数字序号或简单的分行
6. 【转人工提示】当遇到以下情况时，主动建议用户转接人工客服：
   - 复杂的售后问题
   - 需要个人账户信息查询的问题
   - 投诉或纠纷相关问题
   - 知识库无法覆盖的专业技术问题

请基于以上要求，为用户提供专业的客服回答："""

        logger.info("评测器初始化完成")

    def collect_single(self, question: str) -> Dict[str, Any]:
        """对单个问题执行完整的 RAG 管线，采集评测所需数据。

        返回 dict:
            - answer: LLM 生成的回答
            - contexts: 检索到的文档全文列表（full page_content）
            - retrieved_docs_meta: 检索文档的元数据
            - success: 是否成功
            - error: 错误信息（如有）
        """
        try:
            # 第一步：向量检索（使用 MilvusClient，绕过 langchain_milvus bug）
            initial_k = 10 if self.use_rerank and self.rerank_service else 3
            query_vector = self.embeddings.embed_query(question)
            search_results = self._milvus_client.search(
                collection_name=self.Config.MILVUS_COLLECTION_NAME,
                data=[query_vector],
                limit=initial_k,
                output_fields=["content", "doc_name", "doc_path_name", "doc_type"],
                search_params={
                    "metric_type": "COSINE",
                    "params": {"nprobe": 64},
                },
            )

            # 将搜索结果转为 Document 兼容格式
            class _Doc:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata

            # MilvusClient.search 返回 [[Hit, ...]] 外层是 query 列表（这里只有1个query）
            hits = search_results[0] if search_results else []
            candidate_docs = [
                _Doc(hit["entity"]["content"], {
                    "doc_name": hit["entity"].get("doc_name", "unknown"),
                    "doc_path_name": hit["entity"].get("doc_path_name", ""),
                    "doc_type": hit["entity"].get("doc_type", "unknown"),
                    "rerank_score": hit.get("distance", 0),
                })
                for hit in hits
            ]
            logger.info(f"  检索到 {len(candidate_docs)} 个候选文档")

            # 第二步：重排序
            final_docs = candidate_docs
            if self.use_rerank and self.rerank_service and len(candidate_docs) > 1:
                doc_contents = [doc.page_content for doc in candidate_docs]
                rerank_top_n = min(5, len(candidate_docs))

                rerank_response = self.rerank_service.rerank_documents(
                    query=question,
                    documents=doc_contents,
                    top_n=rerank_top_n,
                )

                if rerank_response.success:
                    reranked_docs = []
                    for rerank_doc in rerank_response.documents:
                        if 0 <= rerank_doc.index < len(candidate_docs):
                            original_doc = candidate_docs[rerank_doc.index]
                            original_doc.metadata["rerank_score"] = rerank_doc.relevance_score
                            reranked_docs.append(original_doc)

                    # 安全检查：保留原始最高相似度文档
                    if candidate_docs and len(reranked_docs) > 0:
                        first_doc = candidate_docs[0]
                        first_in_rerank = any(
                            d.metadata.get("doc_name") == first_doc.metadata.get("doc_name")
                            and d.page_content == first_doc.page_content
                            for d in reranked_docs
                        )
                        if not first_in_rerank:
                            first_doc.metadata["rerank_score"] = 1.0
                            reranked_docs.insert(0, first_doc)

                    final_docs = reranked_docs[:3]
                    logger.info(f"  重排序完成，保留 {len(final_docs)} 个文档")
                else:
                    logger.warning(f"  重排序失败，使用原始检索结果")

            # 第三步：构建完整上下文（含文档名标记，供引用溯源使用）
            context_parts = []
            for doc in final_docs:
                doc_name = doc.metadata.get("doc_name", "未知文档")
                context_parts.append(f"【来源：《{doc_name}》】\n{doc.page_content}")
            contexts = context_parts

            # 第四步：调用 LLM 生成回答
            context_text = "\n\n".join(contexts)
            prompt = self.PROMPT_TEMPLATE.format(context=context_text, question=question)

            _prompt_tokens = _completion_tokens = _total_tokens = 0
            with self.get_openai_callback() as cb:
                answer = self.llm.invoke(prompt).content
                _prompt_tokens = cb.prompt_tokens
                _completion_tokens = cb.completion_tokens
                _total_tokens = cb.total_tokens

            # 采集元数据
            retrieved_meta = []
            retrieved_file_set = []
            for doc in final_docs:
                doc_name = doc.metadata.get("doc_name", "unknown")
                retrieved_meta.append({
                    "doc_name": doc_name,
                    "doc_path_name": doc.metadata.get("doc_path_name", ""),
                    "rerank_score": doc.metadata.get("rerank_score"),
                    "content_length": len(doc.page_content),
                })
                retrieved_file_set.append(doc_name)

            # 提取回答中的引用
            cited_files = extract_citations(answer)

            return {
                "answer": answer,
                "contexts": contexts,
                "retrieved_docs_meta": retrieved_meta,
                "retrieved_file_set": retrieved_file_set,
                "cited_files": cited_files,
                "token_usage": {
                    "prompt_tokens": _prompt_tokens,
                    "completion_tokens": _completion_tokens,
                    "total_tokens": _total_tokens,
                },
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"  采集失败: {e}")
            return {
                "answer": "",
                "contexts": [],
                "retrieved_docs_meta": [],
                "token_usage": None,
                "success": False,
                "error": str(e),
            }

    def run_evaluation(
        self,
        qa_pairs: List[Dict[str, Any]],
        output_path: str = "eval_results.json",
    ) -> Dict[str, Any]:
        """执行完整评测流程。

        Args:
            qa_pairs: 评测数据集中的 QA 对列表
            output_path: 结果输出路径

        Returns:
            评测结果字典
        """
        logger.info(f"{'='*60}")
        logger.info(f"开始评测：共 {len(qa_pairs)} 条 QA，rerank={self.use_rerank}")
        logger.info(f"{'='*60}")

        # ── 第一阶段：逐条采集 RAG 管线输出 ──────────────────
        eval_records = []
        for i, qa in enumerate(qa_pairs):
            qid = qa.get("id", f"q{i+1:03d}")
            question = qa["question"]
            ground_truth = qa["ground_truth"]
            q_type = qa.get("question_type", "未知")

            logger.info(f"[{i+1}/{len(qa_pairs)}] {qid} ({q_type}): {question[:50]}...")

            t0 = time.time()
            result = self.collect_single(question)
            elapsed = time.time() - t0

            record = {
                "id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "question_type": q_type,
                "source_doc": qa.get("source_doc", ""),
                "answer": result["answer"],
                "contexts": result["contexts"],
                "retrieved_docs_meta": result["retrieved_docs_meta"],
                "cited_files": result.get("cited_files", []),
                "success": result["success"],
                "error": result["error"],
                "elapsed_seconds": round(elapsed, 2),
                "token_usage": result.get("token_usage"),
            }
            eval_records.append(record)

            if result["success"]:
                logger.info(
                    f"  -> OK ({elapsed:.1f}s, "
                    f"{len(result['contexts'])} docs, "
                    f"{result['token_usage']['total_tokens']} tokens)"
                )
            else:
                logger.warning(f"  -> FAIL: {result['error']}")

            # 请求间隔，避免触发 API 限流
            time.sleep(0.5)

        # ── 第二阶段：Ragas 评测 ──────────────────────────────
        successful_records = [r for r in eval_records if r["success"]]
        failed_records = [r for r in eval_records if not r["success"]]

        logger.info(f"\n采集完成：成功 {len(successful_records)}，失败 {len(failed_records)}")

        ragas_results = {}
        if successful_records:
            logger.info("正在运行 Ragas 评测...")

            # 构建 Ragas 评测数据集
            eval_data = [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": r["contexts"],
                    "ground_truth": r["ground_truth"],
                }
                for r in successful_records
            ]
            hf_dataset = HFDataset.from_list(eval_data)

            # 定义评测指标
            metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall(),
            ]

            # 执行评测
            try:
                ragas_output = ragas_evaluate(
                    dataset=hf_dataset,
                    metrics=metrics,
                    llm=self.ragas_llm,
                    embeddings=self.judge_embeddings,
                )
                # ragas 0.4.x uses _scores_dict (dict of lists), not to_dict()
                ragas_results = ragas_output._scores_dict

                # 计算均值
                metric_scores = {}
                for metric_name, scores in ragas_results.items():
                    if isinstance(scores, list):
                        valid_scores = [s for s in scores if s is not None]
                        if valid_scores:
                            metric_scores[metric_name] = round(
                                sum(valid_scores) / len(valid_scores), 4
                            )

                logger.info(f"\n{'─'*40}")
                logger.info("Ragas 评测结果：")
                for name, score in metric_scores.items():
                    logger.info(f"  {name}: {score}")
                logger.info(f"{'─'*40}")

            except Exception as e:
                logger.error(f"Ragas 评测执行失败: {e}")
                logger.error("请检查 Judge LLM 是否正常，或尝试安装/更新 ragas: pip install -U ragas")
                metric_scores = {}

        # ── 第三阶段：汇总并保存结果 ──────────────────────────
        # ── 第四阶段：引用溯源指标计算 ────────────────────────
        citation_metrics_results = {}
        if successful_records:
            cit_recalls, cit_precisions, cit_accuracies, cit_has = [], [], [], []
            for r in successful_records:
                cited = r.get("cited_files", [])
                retrieved = [d["doc_name"] for d in r.get("retrieved_docs_meta", [])]
                gt_file = r.get("source_doc", "")
                cm = compute_citation_metrics(
                    answer=r["answer"],
                    cited_files=cited,
                    retrieved_files=retrieved,
                    ground_truth_file=gt_file,
                )
                # 将指标注入 record，便于人工审查
                r["citation_metrics"] = cm
                cit_has.append(cm["has_citation"])
                if cm["has_citation"]:
                    cit_recalls.append(cm["citation_recall"])
                    cit_precisions.append(cm["citation_precision"])
                cit_accuracies.append(cm["citation_accuracy"])

            citation_metrics_results = {
                "citation_rate": round(sum(cit_has) / len(cit_has), 4) if cit_has else 0.0,
                "citation_recall": round(sum(cit_recalls) / len(cit_recalls), 4) if cit_recalls else 0.0,
                "citation_precision": round(sum(cit_precisions) / len(cit_precisions), 4) if cit_precisions else 0.0,
                "citation_accuracy": round(sum(cit_accuracies) / len(cit_accuracies), 4) if cit_accuracies else 0.0,
            }

            logger.info(f"\n{'─'*40}")
            logger.info("引用溯源评测结果：")
            logger.info(f"  引用率（Citation Rate）:      {citation_metrics_results['citation_rate']}")
            logger.info(f"  引用召回率（Citation Recall）: {citation_metrics_results['citation_recall']}")
            logger.info(f"  引用精确率（Citation Precision）:{citation_metrics_results['citation_precision']}")
            logger.info(f"  引用准确率（Citation Accuracy）:{citation_metrics_results['citation_accuracy']}")
            logger.info(f"{'─'*40}")
        final_results = {
            "metadata": {
                "eval_time": datetime.now().isoformat(),
                "total_qa": len(qa_pairs),
                "successful": len(successful_records),
                "failed": len(failed_records),
                "use_rerank": self.use_rerank,
            },
            "ragas_scores": metric_scores if successful_records else {},
            "citation_metrics": citation_metrics_results,
            "per_question_results": eval_records,
        }

        # 按 question_type 分组统计
        if metric_scores and successful_records:
            type_groups = {}
            for r in successful_records:
                q_type = r.get("question_type", "未知")
                if q_type not in type_groups:
                    type_groups[q_type] = []
                type_groups[q_type].append(r)

            type_stats = {}
            for q_type, records in type_groups.items():
                type_stats[q_type] = {
                    "count": len(records),
                    "avg_elapsed": round(
                        sum(r["elapsed_seconds"] for r in records) / len(records), 2
                    ),
                }
            final_results["by_question_type"] = type_stats

        # 保存到文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        logger.info(f"\n结果已保存到: {output_path}")

        # 控制台摘要
        logger.info(f"\n{'='*60}")
        logger.info("评测摘要")
        logger.info(f"{'='*60}")
        logger.info(f"  数据集规模: {len(qa_pairs)} 条 QA")
        logger.info(f"  成功采集:   {len(successful_records)} 条")
        logger.info(f"  采集失败:   {len(failed_records)} 条")
        if metric_scores:
            for name, score in metric_scores.items():
                logger.info(f"  {name}: {score}")
        if citation_metrics_results:
            logger.info(f"  引用率（Citation Rate）:      {citation_metrics_results['citation_rate']}")
            logger.info(f"  引用召回率（Citation Recall）: {citation_metrics_results['citation_recall']}")
            logger.info(f"  引用精确率（Citation Precision）:{citation_metrics_results['citation_precision']}")
            logger.info(f"  引用准确率（Citation Accuracy）:{citation_metrics_results['citation_accuracy']}")
        logger.info(f"  结果文件:   {output_path}")
        logger.info(f"{'='*60}")

        return final_results


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PaperPilot RAG 评测脚本（基于 Ragas 框架）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认评测
  python eval_rag.py

  # 指定数据集和输出路径
  python eval_rag.py --dataset eval_dataset.json --output results.json

  # 关闭 rerank 进行对比评测
  python eval_rag.py --no-rerank --output results_no_rerank.json

  # 仅评测前 10 条
  python eval_rag.py --limit 10
        """,
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset.json"),
        help="评测数据集 JSON 文件路径（默认: 同级目录 eval_dataset.json）",
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="评测结果输出路径（默认: eval_results.json）",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="关闭 rerank 进行评测（对比基线）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评测前 N 条 QA（0 表示全部）",
    )
    args = parser.parse_args()

    # 检查 Ragas 是否安装
    if not HAS_RAGAS:
        logger.error("导入评测依赖失败:")
        logger.error(f"  {_RAGAS_IMPORT_ERROR}")
        sys.exit(1)

    # 加载评测数据集
    logger.info(f"加载评测数据集: {args.dataset}")
    try:
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logger.error(f"找不到评测数据集: {args.dataset}")
        logger.error("请确认 eval_dataset.json 文件存在，或使用 --dataset 指定正确路径")
        sys.exit(1)

    qa_pairs = dataset.get("qa_pairs", [])
    total_papers = len(dataset.get("papers", []))
    logger.info(f"数据集信息: {total_papers} 篇文献, {len(qa_pairs)} 条 QA")

    # 限制评测数量
    if args.limit > 0:
        qa_pairs = qa_pairs[: args.limit]
        logger.info(f"限制评测数量: {len(qa_pairs)} 条")

    if not qa_pairs:
        logger.error("评测数据集为空，无法执行评测")
        sys.exit(1)

    # 初始化评测器并执行
    use_rerank = not args.no_rerank
    evaluator = RAGEvaluator(use_rerank=use_rerank)
    evaluator.run_evaluation(qa_pairs=qa_pairs, output_path=args.output)


if __name__ == "__main__":
    main()
