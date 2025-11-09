# 今日のメッセージ
# llmをランダムに選択して一言生成
# azureも選択肢に入れたい
import datetime
from itertools import chain
from pyexpat import model
import re
import time
from langchain_chroma import Chroma
from ollama import chat
from typer import prompt
from pathlib import Path

from wintermute.main import RerankingWithQueryExpansionRetriever, RuriEmbeddingWithPrefixV3, extract_response_text, format_context, remove_think_tags, save_chat_history_markdown
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

def today_message():
    import random
    
    llm_choices = [
        "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M",
        "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M",
        "hf.co/mmnga/ELYZA-Shortcut-1.0-Qwen-32B-gguf:Q4_K_M",
        "qwen3:14b",
        "gpt-oss:20b",
        "qwen3:30b",
        "hf.co/mmnga/deepseek-r1-distill-qwen2.5-bakeneko-32b-gguf:Q4_K_M",
        "hf.co/mmnga/qwen2.5-bakeneko-32b-instruct-v2-gguf:Q4_K_M",
        "hf.co/tensorblock/DeepSeek-R1-Distill-Qwen-14B-Japanese-GGUF:Q6_K",
        "gemma3:27b",
        "hf.co/gabriellarson/Tongyi-DeepResearch-30B-A3B-GGUF:Q4_K_M",
        "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q8_0",
        "hf.co/Qwen/Qwen3-8B-GGUF:Q8_0",
        "hf.co/LiquidAI/LFM2-2.6B-GGUF:F16",
        "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:F16",
        "deepseek-v3.1:671b-cloud", # たぶん期間限定
        "gpt-oss:120b-cloud"    # たぶん期間限定
    ]
    selected_model = random.choice(llm_choices)
    dir_path = Path(r"C:/Users/teleg/OneDrive/0notes")
    llm = OllamaLLM(model=selected_model)
    # llm = OllamaLLM(model = "qwen3:32b")
    llm2 = OllamaLLM(model = selected_model)
    embedding = RuriEmbeddingWithPrefixV3(device="cuda")
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )

    print(f"今日の言葉を考えています。: {selected_model}")
    print(vectordb._collection.count())

    base_retriever = vectordb.as_retriever(
        search_kwargs={"k": 10}
    )
    reranker = RerankingWithQueryExpansionRetriever(
        base_retriever=base_retriever,
        llm=llm,
        top_k=10,
        embedding=embedding,
        query_expansion_llm=llm2
    )

    datestr = "今は " + datetime.datetime.now().strftime("%Y年%m月%d日 %H時%M分") + " です。"
    query = datestr + " " + "今日の言葉を日本語で生成してください。その言葉を選んだ理由も教えてください。"
    prompt_template = PromptTemplate.from_template(
        """あなたは、私の利益を最大化するように導くことを目的としたai、wintermuteです。
        以下の文書をもとに分析し、あなたの知識も使って判断し回答してください。
        '#continue'タグの内容も参考にしてください
        文書の[日時]は情報が記録された日時として重要です。可能な限り時系列で推論してください。
        助言を求められた場合は、目的に向かって正しく導くよう努めてください。

        検索文書: {context}
        
        検索クエリ: {question}
        
        今日の言葉:"""
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=reranker,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )
    start_answer = time.perf_counter()
    context_docs = qa.retriever.invoke(query)
    context = format_context(context_docs)
    response = llm.invoke(prompt_template.format(context=context, question=query))
    response_text = extract_response_text(response)
    response_text = remove_think_tags(response_text)
    end_answer = time.perf_counter()
    print(f"今日の言葉: {response_text}\n{llm.model}")
    print(f"回答生成時間: {end_answer - start_answer:.2f}秒")
    fsave = prompt("このメッセージを保存しますか？ (y/n)", default="n")
    if fsave.lower() == "y":
        save_chat_history_markdown(chat_history=[(f"今日の言葉 by {llm.model}",response_text)], vaultdir=dir_path)
    return

# テスト実行
if __name__ == "__main__":
    today_message()