"""
快速 Ragas 评测 - 直接从 eval_results.json 读取已有数据，跳过采集阶段

用法: python compute_ragas.py
"""
import json
import re
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("compute_ragas")

# vertexai stub（ragas 兼容性）
import types
def _patch_vertexai_stub():
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

# 导入 ragas
try:
    from datasets import Dataset as HFDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from openai import OpenAI
except ImportError as e:
    logger.error(f"导入失败: {e}")
    sys.exit(1)

# 加载 eval_results.json
input_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
if not os.path.exists(input_path):
    logger.error(f"找不到 {input_path}，请先运行 eval_rag.py 生成数据")
    sys.exit(1)

with open(input_path, "r", encoding="utf-8") as f:
    results = json.load(f)

records = results.get("per_question_results", [])
successful = [r for r in records if r.get("success")]
failed = [r for r in records if not r.get("success")]

logger.info(f"读取到 {len(successful)} 条成功记录，{len(failed)} 条失败记录")

if not successful:
    logger.error("没有成功采集的数据，无法评测")
    sys.exit(1)

# 初始化 LLM 和 Embedding
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wxkf_dir = os.path.join(project_root, "mildoc_wxkf")
sys.path.insert(0, wxkf_dir)
from config import Config

logger.info("初始化 LLM...")
judge_llm = ChatOpenAI(
    model=Config.LLM_MODEL_NAME,
    api_key=Config.LLM_API_KEY,
    base_url=Config.LLM_BASE_URL,
    temperature=0.1,
)

class CustomEmbeddings:
    def __init__(self, model_name, api_key, api_base, dimensions):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.dimensions = dimensions
    def embed_query(self, text): return self.embed_documents([text])[0]
    def embed_documents(self, texts):
        r = self.client.embeddings.create(model=self.model_name, input=texts, dimensions=self.dimensions, encoding_format="float")
        return [d.embedding for d in r.data]

logger.info("初始化 Embedding...")
embeddings = CustomEmbeddings(
    model_name=Config.LLM_EMBEDDING_MODEL_NAME,
    api_key=Config.LLM_EMBEDDING_API_KEY,
    api_base=Config.LLM_EMBEDDING_BASE_URL,
    dimensions=Config.MILVUS_VECTOR_DIM,
)

ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# 构建 Ragas 数据集
logger.info("构建评测数据集...")
eval_data = [
    {
        "question": r["question"],
        "answer": r["answer"],
        "contexts": r["contexts"],
        "ground_truth": r["ground_truth"],
    }
    for r in successful
]
hf_dataset = HFDataset.from_list(eval_data)

# 分开跑指标：context_precision 必须单独跑，否则会报 nan（ragas bug）
logger.info("阶段1：评测 faithfulness / answer_relevancy / context_recall...")
output1 = ragas_evaluate(
    dataset=hf_dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)
results1 = output1._scores_dict

logger.info("阶段2：单独评测 context_precision（避免 ragas 内部 callback 冲突）...")
output2 = ragas_evaluate(
    dataset=hf_dataset,
    metrics=[ContextPrecision()],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)
results2 = output2._scores_dict

# 合并结果
ragas_results = {**results1, **results2}

# 计算均值
metric_scores = {}
for metric_name, scores in ragas_results.items():
    valid = [s for s in scores if s is not None and not (isinstance(s, float) and s != s)]
    if valid:
        metric_scores[metric_name] = round(sum(valid) / len(valid), 4)

logger.info(f"\n{'='*50}")
logger.info("Ragas 评测结果：")
for name, score in metric_scores.items():
    logger.info(f"  {name}: {score}")
logger.info(f"{'='*50}")

# 合并到原结果并保存
results["ragas_scores"] = metric_scores

# ── 引用溯源指标重算 ──────────────────────────────────────
def _extract_citations(answer):
    pattern = r"【来源：《([^》]+)》】"
    matches = re.findall(pattern, answer)
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def _compute_citation_metrics(answer, cited_files, retrieved_files, ground_truth_file):
    has_citation = len(cited_files) > 0
    citation_recall = 1.0 if ground_truth_file in cited_files else 0.0
    if cited_files:
        cited_in_retrieved = sum(1 for f in cited_files if f in retrieved_files)
        citation_precision = round(cited_in_retrieved / len(cited_files), 4)
    else:
        citation_precision = 0.0
    citation_accuracy = 1.0 if ground_truth_file in cited_files else 0.0
    return {
        "has_citation": has_citation,
        "cited_files": cited_files,
        "citation_recall": citation_recall,
        "citation_precision": citation_precision,
        "citation_accuracy": citation_accuracy,
    }


if successful:
    cit_recalls, cit_precisions, cit_accuracies, cit_has = [], [], [], []
    for r in successful:
        cited = _extract_citations(r.get("answer", ""))
        retrieved = [d["doc_name"] for d in r.get("retrieved_docs_meta", [])]
        gt_file = r.get("source_doc", "")
        cm = _compute_citation_metrics(r.get("answer", ""), cited, retrieved, gt_file)
        r["citation_metrics"] = cm
        r["cited_files"] = cited
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
    results["citation_metrics"] = citation_metrics_results

    logger.info(f"\n{'─'*40}")
    logger.info("引用溯源评测结果：")
    logger.info(f"  引用率（Citation Rate）:      {citation_metrics_results['citation_rate']}")
    logger.info(f"  引用召回率（Citation Recall）: {citation_metrics_results['citation_recall']}")
    logger.info(f"  引用精确率（Citation Precision）:{citation_metrics_results['citation_precision']}")
    logger.info(f"  引用准确率（Citation Accuracy）:{citation_metrics_results['citation_accuracy']}")
    logger.info(f"{'─'*40}")

with open(input_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

logger.info(f"结果已更新到 {input_path}")
