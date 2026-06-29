"""
RAG 评测脚本 - 基于 Ragas 框架

使用 Ragas 框架对 PaperPilot 的 RAG 管线进行系统评测，
覆盖 Faithfulness / Answer Relevancy / Context Precision / Context Recall 四项指标。

运行方式：
    1. 确保 mildoc_wxkf 服务已配置（.env 文件中 LLM、Embedding、Milvus 等配置正确）
    2. 将待评测的文献上传到 MinIO 知识库（确保 mildoc_index 已完成索引）
    3. 在 mildoc_wxkf 目录下执行：
       python eval_rag.py
    4. 可选参数：
       --dataset ../eval_dataset.json    评测数据集路径（默认同级目录下的 eval_dataset.json）
       --output eval_results.json        结果输出路径（默认 eval_results.json）
       --no-rerank                       关闭 rerank 进行评测
       --limit N                         仅评测前 N 条 QA

依赖安装：
    pip install ragas datasets langchain-openai langchain-community
"""

import os
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
# 延迟导入 Ragas（首次运行前会自动检查）
# ─────────────────────────────────────────────────────────────
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
except ImportError:
    HAS_RAGAS = False


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

        # 延迟导入项目内部模块（需要在 mildoc_wxkf 目录下运行）
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from config import Config
        from rag_service import RAGService
        from langchain_openai import ChatOpenAI
        from langchain_milvus import Milvus

        self.Config = Config
        self.RAGService = RAGService

        # 初始化 RAG 服务（复用已有的 Milvus 连接、Embedding、LLM）
        logger.info("正在初始化 RAG 服务...")
        self.rag_service = RAGService()

        # 单独初始化一个用于 Ragas 评判的 LLM（Judge LLM）
        # 复用同一个 LLM 配置，但独立实例以避免回调冲突
        logger.info("正在初始化 Ragas Judge LLM...")
        self.judge_llm = ChatOpenAI(
            model=Config.LLM_MODEL_NAME,
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            temperature=0.1,
        )

        # Ragas 需要的 Embedding 模型包装器
        self.judge_embeddings = LangchainEmbeddingsWrapper(self.rag_service.embeddings)

        # Ragas LLM 包装器
        self.ragas_llm = LangchainLLMWrapper(self.judge_llm)

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
            # 第一步：向量检索（与 query_service 逻辑一致，但保留完整 page_content）
            initial_k = 10 if self.use_rerank and self.rag_service.rerank_service else 3
            retriever = self.rag_service.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": initial_k},
            )
            candidate_docs = retriever.invoke(question)
            logger.info(f"  检索到 {len(candidate_docs)} 个候选文档")

            # 第二步：重排序
            final_docs = candidate_docs
            if (
                self.use_rerank
                and self.rag_service.rerank_service
                and len(candidate_docs) > 1
            ):
                doc_contents = [doc.page_content for doc in candidate_docs]
                rerank_top_n = min(5, len(candidate_docs))

                rerank_response = self.rag_service.rerank_service.rerank_documents(
                    query=question,
                    documents=doc_contents,
                    top_n=rerank_top_n,
                )

                if rerank_response.success:
                    reranked_docs = []
                    for rerank_doc in rerank_response.documents:
                        if 0 <= rerank_doc.index < len(candidate_docs):
                            original_doc = candidate_docs[rerank_doc.index]
                            if hasattr(original_doc, "metadata"):
                                original_doc.metadata["rerank_score"] = rerank_doc.relevance_score
                            reranked_docs.append(original_doc)

                    # 安全检查：保留原始最高相似度文档
                    if candidate_docs and len(reranked_docs) > 0:
                        first_doc = candidate_docs[0]
                        first_in_rerank = any(
                            hasattr(d, "metadata")
                            and d.metadata.get("doc_name") == first_doc.metadata.get("doc_name")
                            and d.page_content == first_doc.page_content
                            for d in reranked_docs
                        )
                        if not first_in_rerank:
                            if hasattr(first_doc, "metadata"):
                                first_doc.metadata["rerank_score"] = 1.0
                            reranked_docs.insert(0, first_doc)

                    final_docs = reranked_docs[:3]
                    logger.info(f"  重排序完成，保留 {len(final_docs)} 个文档")
                else:
                    logger.warning(f"  重排序失败，使用原始检索结果")

            # 第三步：构建完整上下文（**关键：使用完整 page_content，不截断**）
            contexts = [doc.page_content for doc in final_docs]

            # 第四步：调用 LLM 生成回答
            from langchain_community.callbacks.manager import get_openai_callback

            context_text = "\n\n".join(contexts)
            prompt = self.rag_service.PROMPT_TEMPLATE.format(
                context=context_text, question=question
            )

            with get_openai_callback() as cb:
                answer = self.rag_service.llm.invoke(prompt).content

            # 采集元数据
            retrieved_meta = []
            for doc in final_docs:
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                retrieved_meta.append({
                    "doc_name": meta.get("doc_name", "unknown"),
                    "doc_path_name": meta.get("doc_path_name", ""),
                    "rerank_score": meta.get("rerank_score"),
                    "content_length": len(doc.page_content),
                })

            return {
                "answer": answer,
                "contexts": contexts,
                "retrieved_docs_meta": retrieved_meta,
                "token_usage": {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
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
                "answer": result["answer"],
                "contexts": result["contexts"],
                "retrieved_docs_meta": result["retrieved_docs_meta"],
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
                ragas_results = ragas_output.to_dict()

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
        final_results = {
            "metadata": {
                "eval_time": datetime.now().isoformat(),
                "total_qa": len(qa_pairs),
                "successful": len(successful_records),
                "failed": len(failed_records),
                "use_rerank": self.use_rerank,
            },
            "ragas_scores": metric_scores if successful_records else {},
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
  python eval_rag.py --dataset ../eval_dataset.json --output results.json

  # 关闭 rerank 进行对比评测
  python eval_rag.py --no-rerank --output results_no_rerank.json

  # 仅评测前 10 条
  python eval_rag.py --limit 10
        """,
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_dataset.json"),
        help="评测数据集 JSON 文件路径（默认: ../eval_dataset.json）",
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
        logger.error("缺少评测依赖，请先安装:")
        logger.error("  pip install ragas datasets langchain-openai langchain-community")
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
