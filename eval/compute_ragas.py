"""
快速 Ragas 评测 - 直接从 eval_results.json 读取已有数据，跳过采集阶段

用法: python compute_ragas.py
"""
import json
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
with open(input_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

logger.info(f"结果已更新到 {input_path}")
