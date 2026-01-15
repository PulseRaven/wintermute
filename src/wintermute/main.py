# winternute ver2.8
# Wintermuteは、ObsidianのVaultをベクトルDBに登録して、自然言語で質問できるようにするツールです。
# splitterを選択できるようにしました。(v2.8)

from asyncio.windows_utils import pipe
from email.policy import default
from functools import cache
from hmac import new
from math import e, exp
import random
import re
import token
from tracemalloc import start
from turtle import goto
from unittest import result
from unittest.mock import Base
from xml.etree.ElementInclude import include
from click import prompt
from httpx import get
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from pathlib import Path
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.docstore.document import Document
from numpy import isin
from regex import F
from sqlalchemy import all_
from sympy import im, rem, use
import transformers
# from wintermute.today_msg import today_message
from wintermute.utils.custom_direcotory_loader import CustomDirectoryLoader
from sentence_transformers import SentenceTransformer
from langchain.schema import BaseRetriever, Document
from typing import List, Any
import torch
torch.backends.cudnn.benchmark = True
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
import asyncio
from pydantic import PrivateAttr
from dotenv import load_dotenv

load_dotenv()

import os
import shutil
import pickle
import time
import gc
import sys
import datetime
import json
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
import numpy as np
from langchain_community.llms import HuggingFacePipeline

def main():
    # 循環インポートの問題をクリアするまでは単独実行にする
    # today_msg = (input("今日の言葉を生成しますか？(y/n): ").strip().lower() == 'y')
    # if today_msg:
    #     today_message()
    #     return

    # llm2 = OllamaLLM(model = "ministral-3:14b")  軽いわりにクエリ拡張が優秀だが、2巡目になるととたんに遅くなる
    llm2 = OllamaLLM(model = "qwen3:32b")
    # llm2 = OllamaLLM(model = "hf.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M") #3090なし運用 
    # llm2 = AzureAIChatCompletionsModel(
    #     endpoint = os.getenv("AZURE_DEEPSEEK_ENDPOINT"),
    #     credential= os.getenv("AZURE_DEEPSEEK_CREDENTIAL"),
    #     model = "deepseek-v3.1"
    # )
    # print(f"Using deepseek-v3.1 on Azure AI as llm2")
    # llm2 = AzureAIChatCompletionsModel(
    #     endpoint = os.getenv("AZURE_GPT41_ENDPOINT"),
    #     credential = os.getenv("AZURE_GPT41_CREDENTIAL"),
    #     model = "gpt-4.1"
    # )
    # print(f"Using gpt-4.1 on Azure AI as llm2")

    # llm2 = OllamaLLM(model = "qwen3:32b")  今のところローカルでベストだと思う
    # llm2 = OllamaLLM(model = "hf.co/unsloth/aquif-3.5-Max-42B-A3B-GGUF:Q4_K_M") クエリ拡張で返ってこなくなることがある
    # llm2 = OllamaLLM(model = "mistral-small3.2:latest") 悪くないがcpuオフロードで重い
    print(f"Using {llm2.model} on Ollama as llm2")

    # gpu確認
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

     # 1. ベクトルDBの永続化ディレクトリとObsidian Vaultのパス設定
    reindex = True  # Trueにすると再インデックス。
    use_cloudllm = False  # TrueにするとAzure OpenAIを使う。FalseだとOllamaのローカルモデル
    target_vault = input("Target vault? ([0]notes/[v]scodenote) [default: 0]: ").strip()
    if not target_vault:
        target_vault = "0"
    if target_vault == "0":
        target_vault = "0notes"
    elif target_vault == "v":
        target_vault = "vscodenote"
    if target_vault == "0notes":
        persist_directory = "chroma_db"
        dir_path = Path(r"C:/Users/teleg/OneDrive/0notes")
    elif target_vault == "vscodenote":
        persist_directory = "chroma_db_vs"
        dir_path = Path(r"C:/Users/teleg/OneDrive - 東海銑鉄株式会社/vscodenote")
    else:
        raise ValueError(f"Unknown vault: {target_vault}")
    

    print(f"initialize wintermute...({target_vault})")
    models = [
        "qwen3:14b",
        "qwen3:30b",
        "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M",
        "hf.co/mmnga/qwen2.5-bakeneko-32b-instruct-v2-gguf:Q4_K_M",
        "hf.co/mmnga/deepseek-r1-distill-qwen2.5-bakeneko-32b-gguf:Q4_K_M",
        "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M",
        "hf.co/mmnga/ELYZA-Shortcut-1.0-Qwen-32B-gguf:Q4_K_M",
        "qwen3:32b",
        "gpt-oss:20b", 
        "hf.co/gabriellarson/Tongyi-DeepResearch-30B-A3B-GGUF:Q4_K_M",
        "gemma3:27b",
        "Qwen3:8B",
        # "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:F16",
        "qwen3-vl:235b-cloud",
        "deepseek-v3.1:671b-cloud", # トークン制限あり
        "deepseek-v3.2:cloud",
        "hf.co/unsloth/aquif-3.5-Max-42B-A3B-GGUF:Q4_K_M",
        "glm-4.6:cloud",
        "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q4_K_M",
        # "mistral-small3.2:latest",
        "hf.co/mradermacher/GLM-4-32B-0414-GGUF:Q4_K_M",
        "deepseek-r1:32b"
    ]
    # 優秀なクラウドモデル群
    first_models = [
        "gpt-4.1",
        "gpt-5.2-chat",
        # "deepseek-v3.1:671b-cloud", 
        "deepseek-v3.2:cloud", 
        # "qwen3-vl:235b-cloud", 結構落ちるので外す
        # "glm-4.6:cloud",
        "glm-4.7:cloud",
        "qwen3-next:80b-cloud",
        "mistral-large-3:675b-cloud"
    ]

    # 会話履歴
    chat_history = []
    history_max = 5
    first_question = None  # 最初の質問をホールドする変数

    randomMode = False  # Ensure randomMode is always defined
    use_cloudllm = input("Use Azure AI? (y/n) [default: n]: ").strip().lower() == 'y'
    if use_cloudllm:
        use_gpt41 = input("Use GPT-4.1? (y/n) [default: n]: ").strip().lower() == 'y'
        if use_gpt41:
            print("Using gpt-4.1 on Azure AI")
            llm = AzureAIChatCompletionsModel(
                endpoint = os.getenv("AZURE_GPT41_ENDPOINT"),
                credential = os.getenv("AZURE_GPT41_CREDENTIAL"),
                model = "gpt-4.1"
            )
            model_name = "gpt-4.1 on Azure AI"
        else:
            print("Using gpt-5.2-chat on Azure AI")
            llm = AzureAIChatCompletionsModel(
                endpoint = os.getenv("AZURE_GPT52CHAT_ENDPOINT"),
                credential= os.getenv("AZURE_GPT52CHAT_CREDENTIAL"),
                model = "gpt-5.2-chat"
            )
            # model_name = "deepseek-v3.1 on Azure AI"
            # print("Using Deepseek-V3.1 on Azure AI")
            # llm = AzureAIChatCompletionsModel(
            #     endpoint = os.getenv("AZURE_DEEPSEEK_ENDPOINT"),
            #     credential= os.getenv("AZURE_DEEPSEEK_CREDENTIAL"),
            #     model = "deepseek-v3.1"
            # )
            # model_name = "deepseek-v3.1 on Azure AI"

        randomMode = False
    else:
        if input("Random LLM Model? (y/n) [default: n]: ").strip().lower() == 'y':
            randomMode = True
            llm = random_model(first_models)
        else:
            if input("Use Feature model? (y/n) [default: n]: ").strip().lower() == 'y':
                # llm = OllamaLLM(model = "qwen3:4b") # テスト用
                llm = OllamaLLM(model = "qwen3-next:80b-cloud")
                # llm = OllamaLLM(model = "hf.co/unsloth/aquif-3.5-Max-42B-A3B-GGUF:Q4_K_M")
                # llm = OllamaLLM(model = "nemotron-3-nano")    
                # llm = OllamaLLM(model = "ministral-3:14b")               
                # llm = OllamaLLM(model = "hf.co/mradermacher/GLM-4-32B-0414-GGUF:Q4_K_M")
                # llm = OllamaLLM(model = "hf.co/bartowski/THUDM_GLM-Z1-32B-0414-GGUF:Q4_K_M")
            else:
                llm = OllamaLLM(model = "deepseek-v3.2:cloud")
                # llm = OllamaLLM(model = "hf.co/unsloth/aquif-3.5-Max-42B-A3B-GGUF:Q4_K_M")
                # llm = OllamaLLM(model = "qwen3-vl:235b-cloud")
            # llm = OllamaLLM(model = "gemma3:27b")
            # llm = OllamaLLM(model = "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M")
            # llm = OllamaLLM(model = "hf.co/mmnga/qwen2.5-bakeneko-32b-instruct-v2-gguf:Q4_K_M" )
            # llm = OllamaLLM(model = "hf.co/gabriellarson/Tongyi-DeepResearch-30B-A3B-GGUF:Q4_K_M")
            # ここから下は、vscodenoteの最優先のタスクを見逃してしまったことがあるので注意が必要
            # llm = OllamaLLM(model = "qwen3:30b")
            # llm = OllamaLLM(model = "qwen3:14b")
            # llm = OllamaLLM(model = "hf.co/mmnga/deepseek-r1-distill-qwen2.5-bakeneko-32b-gguf:Q4_K_M")
            # llm = OllamaLLM(model = "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M")
            # llm = OllamaLLM(model = "hf.co/mmnga/ELYZA-Shortcut-1.0-Qwen-32B-gguf:Q4_K_M")
            # llm = OllamaLLM(model = "gpt-oss:20b") 
            # llm = OllamaLLM(model = "qwen3:32b") # cpu offload 重い
            print(f"Using {llm.model} on Ollama")
            randomMode = False

    embedding = RuriEmbeddingWithPrefixV3(device="cuda:1")  # ruri v3専用embeddingラッパー 12GB側（GPU 1）を使う場合 時々入れ替わるので注意が必要
    reindex = input(f"reindex? (y/n) [default: n]: ").strip().lower() == 'y'  # 再インデックスするかどうか
    # 初回 or 再インデックス時
    if reindex :
        # splitterの選択
        splitter_choice = input("Use MarkdownTextSplitter? (y/n) [default: y]: ").strip().lower()
        if not splitter_choice:
            splitter_choice = 'y'
        if splitter_choice == 'y':
            print("Using MarkdownTextSplitter")
            splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        else:
            print("Using RecursiveCharacterTextSplitter")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", "。",  ".", " "])

        include_ai_gen = input(f"Include ai_generated? (y/n) [default: n]: ").strip().lower() == 'y'
        if include_ai_gen:
            exclude_dirs = ["devdoc", "copilot-custom-prompts"]
        else: 
            exclude_dirs = ["devdoc", "ai_generated", "copilot-custom-prompts"]
        # 2. 複数ファイルの読み込み＋チャンク化
        loader = CustomDirectoryLoader(
            str(dir_path), 
            exclude_dirs= exclude_dirs,  # 除外ディレクトリ
            glob="**/*.md", #再帰的に.mdファイルを検索
            loader_cls=TextLoader, 
            loader_kwargs={"encoding": "utf-8"},
            recursive = True
        )
        # loader = TextLoader(r"C:\Users\teleg\OneDrive\0notes\archive\2025年06月02日.md", encoding="utf-8")
        docs = loader.load_exclude()
        for doc in docs:
            # タイムスタンプをmetadataに追加
            # タイムスタンプについては、metatadataをprefilterでフィルタしないとembeddingではかなり弱い
            file_path = doc.metadata.get("source", "")
            if file_path and os.path.exists(file_path):
                timestamp = os.path.getmtime(file_path)
                dt = datetime.datetime.fromtimestamp(timestamp)
                iso_str = dt.isoformat()
                jp_str = dt.strftime("%Y年%m月%d日 %H:%M:%S")
                doc.metadata["timestamp"] = iso_str  # ISOフォーマットで保存
                doc.page_content = f"[日時: {iso_str} / {jp_str} / {dt.strftime('%Y年%m月')}]\n" + doc.page_content

        splits = splitter.split_documents(docs) # timestampがmetadataに含まれているか確認が必要かもしれない
        # デバッグ用: チャンクの内容を確認
        # for i, split in enumerate(splits):
        #     if "#continue" in split.page_content:
        #         print(f"チャンク{i}に#continueあり: {split.page_content[:300]}")
        #     # デバッグ用: 近い文字列も検出
        #     if any(word in split.page_content.lower() for word in ["#continue", "＃continue", "#conti", "#cont", "#conti\n"]):
        #         print(f"類似タグ検出: {split.page_content[:60]}")

        batch_size = 1000
        processed_count = 0
        vectordb = None

        # キャッシュとdbを削除
        print("vectordbを初期化します。")
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        register_to_chroma_batch(
            docs=splits,
            embedding=embedding,
            persist_directory=persist_directory,
            batch_size=batch_size
        )
        print(f"vectordbの初期化完了: {persist_directory}")

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    if vectordb is None:
        print("vectordbの初期化に失敗しました。")
        return

    print(vectordb._collection.count())  # ベクトルDBに登録された件数

    # Chromaからすべてのドキュメントを取得して、最近の要約を作成
    all_docs = vectordb._collection.get(include=["documents", "metadatas"])
    adocs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_docs.get("documents") or [], all_docs.get("metadatas") or [])
    ]
    recent_docs, _ = prefilter_by_timestamp(adocs, after_datetime=datetime.datetime.now() - datetime.timedelta(days=14))  # 過去14日以内のドキュメントを取得
    print(f"Recent docs count: {len(recent_docs)}")
    # recent_docs = summarize_recent_docs(adocs, llm)
    # recent_doc = Document(page_content=recent_summary, metadata={"source": "recent_summary"})
    base_retriever = vectordb.as_retriever(
        search_kwargs ={"k": 15}    # RerankingWithQueryExpansionRetrieverでの取得件数
            # , "score_threshold": 0.2},
            # search_type = "similarity_score_threshold"
    )
    reranker = RerankingWithQueryExpansionRetriever (
        base_retriever=base_retriever,
        reranker_model_name="cl-nagoya/ruri-v3-reranker-310m",  # Ruri v3 reranker
        top_k=50,  
        query_expansion_llm=llm2,  # クエリ拡張用のLLM
        embedding=embedding,  # 埋め込みモデルを渡す
        recent_docs=recent_docs,
        device="cuda:1" 
    )
    while True:
        # 問い合わせ入力
        print("質問を入力してください。ctrl+zで終了\n")
        querybody = sys.stdin.read()
        datestr = "今は " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " です。\n"
        query = datestr + querybody
        print("thinking...")

        # 最初の質問をホールド
        start_summary = time.perf_counter()
        if first_question is None:
            first_question = query

        # 会話履歴の更新
        history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history[-history_max:]])
        if history_str:
            summary_prompt = f"以下の会話履歴を履歴に沿って日本語で要約してください。\n\n{history_str}"
            summary_obj = llm2.invoke(summary_prompt)
            if hasattr(summary_obj, "content"): # AIMessage型ならcontentを取り出す
                summary_obj = summary_obj.content
        else:
            summary_obj = None
        summary_obj = remove_think_tags(summary_obj) if summary_obj else None
        conversational_query = f"最初の質問: {first_question}\n\n履歴要約: {summary_obj}\n\n新しい質問: {query}" if summary_obj else query
        end_summary = time.perf_counter()
        print(f"conversational_query: {conversational_query}\n(要約時間: {end_summary - start_summary:.2f}秒)")

        # 4. 日本語プロンプトの設定
        # 少し緩める
        prompt_template = PromptTemplate.from_template(
            """あなたは、私のためのai wintermuteです。
            以下の与えられた検索文書の情報をもとにあなたの考えを日本語で答えてください
            #ignore, #junkというタグがついている文書は特別な指示がない限り無視してください
            回答にタグを含む場合は、タグをバッククォートしてください

            検索文書: {context}

            検索クエリ: {question}

            回答 :"""
        )

        # 5. 検索QAチェーンを作成
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=reranker,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": prompt_template,
            },
            return_source_documents=True,
        )

        # 6. 質問
        start_answer = time.perf_counter()
        context_docs = qa.retriever.invoke(conversational_query)
        context = format_context(context_docs)
        # response = qa.invoke({"query": conversational_query, "context": context})
        response = llm.invoke(
            prompt_template.format(context=context, question=conversational_query)
        )
        response_text = extract_response_text(response)
        response_text = remove_think_tags(response_text)
        end_answer = time.perf_counter()
        print(response_text)
        import winsound
        winsound.Beep(300, 100)  # 300Hz, 100ms ビープ音
        print(f"回答時間: {end_answer - start_answer:.2f}秒")
#        print(response.get("source_documents"))

        # 会話履歴に追加
        chat_history.append((query, response_text))

        # 再び問い合わせ入力に戻る
        cont = input("続けて質問しますか？(y/n): ")
        if cont.lower() != 'y':
            fsave = input("チャット履歴を保存しますか？(y/n): ")
            if fsave.lower() == 'y':
                if use_cloudllm:
                    llm_name = model_name
                else:
                    llm_name = getattr(llm, 'model', None)
                save_chat_history_markdown(chat_history, dir_path, llm_name=llm_name)
            newchat = input("新しいチャットを始めますか？(y/n): ")
            if newchat.lower() == 'y':
                first_question = None
                chat_history.clear()
            else:
                break
        gc.collect() 
        torch.cuda.empty_cache() #これは結構効いている
        if randomMode:
            # ランダムモードならモデルを変える
            llm = random_model(first_models)

# RerankingRetrieverクラスの定義>
# クエリ拡張を実装
class RerankingWithQueryExpansionRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    top_k: int = 30 #最終的にllmに渡されるドキュメント数
    _device: Any = PrivateAttr()
    reranker_model_name: str = "cl-nagoya/ruri-v3-reranker-310m"
    query_expansion_llm: Any
    embedding: Any
    recent_docs: Any = None  # 近況のドキュメント
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _expansion_prompt: Any = PrivateAttr()

    def __init__(self, **data):   
        super().__init__(**data)
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = data.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_str)
        self._tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self._device).to(torch.float16)
        self.embedding = data.get("embedding")
        self.recent_docs = data.get("recent_docs")
        # クエリ拡張用プロンプト
        self._expansion_prompt = ChatPromptTemplate.from_template(
            """これは、obsidianのvaultに対するRAGのクエリ拡張のためのプロンプトです。
            以下の質問から、RAGのクエリ拡張のための複数の多様な質問を生成してください
            質問の意図をいくつか推測し、それに基づいた質問を生成してください
            元の質問の中にタグが含まれている場合、タグをそのまま残したクエリを最低一つ生成してください。
            元の質問の中の(新しい質問)は、最新の質問です。これを重視したクエリを最低1つ生成してください
            生成する検索クエリは、単純なJSON形式のリストで出力してください。
            例:
            ["クエリ1", "クエリ2", "クエリ3"]
            他の形式（辞書型やオブジェクト型）は絶対に使わないでください。

            元の質問: {query}

            生成するクエリ:
            """
        )

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        start_expand = time.perf_counter()
        expansion_chain = self._expansion_prompt | self.query_expansion_llm
        expanded_queries_output = expansion_chain.invoke({"query": query})
        try:
            # AIMessage型ならcontentを取り出す
            if hasattr(expanded_queries_output, "content"):
                expanded_queries_str = expanded_queries_output.content  # ai_messageの場合 gpt-oss:120b
            else:
                expanded_queries_str = expanded_queries_output
            # <think>...</think>タグを除去
            expanded_queries_str = remove_think_tags(expanded_queries_str)
            # match = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', expanded_queries_str) bakeneko, gpt-oss
            match = re.search(r'\s*(\[[\s\S]*?\])\s*', expanded_queries_str)    #qwen3-30b
            if match:
                expanded_queries = json.loads(match.group(1))
            else:
                print(f"クエリ拡張の結果がJSON形式ではありません: {expanded_queries_str}")
                expanded_queries = [query]  # JSON形式でない場合は元のクエリを使用
        except Exception as e:
            print(f"クエリ拡張に失敗 {e}: {expanded_queries_str}")
            expanded_queries = [query]  # eval失敗時は元のクエリを使用

        print(f"クエリ拡張結果: {expanded_queries}")
        print(f"クエリ拡張時間: {time.perf_counter() - start_expand:.2f}秒")
        # クエリを展開して、各クエリでドキュメントを取得
        all_initial_docs = []
        start_retrieve = time.perf_counter()
        for q in expanded_queries:
            # print(f"Retrieving documents for query: {q}")
            # dict型に対応
            if isinstance(q, dict) and "query" in q:
                query_text = q["query"]
            elif isinstance(q, str):
                query_text = q
            else:
                print(f"不正なクエリ形式: {q}")
                continue
            docs = self.base_retriever.invoke(query_text)
            all_initial_docs.extend(docs)
        # 近況ドキュメントも追加
        if self.recent_docs:
            all_initial_docs.extend(self.recent_docs)
        # 重複を削除
        # unique_initial_docs = list({doc.page_content: doc for doc in all_initial_docs}.values())
        unique_initial_docs = remove_semantic_duplicates_fast(all_initial_docs, self.embedding, threshold=0.95)
        # unique_initial_docs = all_initial_docs
        print(f"取得したドキュメント数: {len(unique_initial_docs)}")
        print(f"ドキュメント取得時間: {time.perf_counter() - start_retrieve:.2f}秒")
        if not unique_initial_docs:
            return []
        
        start_rerank = time.perf_counter()
        # 結合されたドキュメントをrerankerにかける(一括処理)
        # pairs = [(query, doc.page_content) for doc in unique_initial_docs]
        # inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(self._device)
        # with torch.no_grad():
        #     logits = self._model(**inputs).logits
        #     if logits.ndim == 2 and logits.shape[1] == 1:
        #         scores = logits.squeeze(-1)
        #     else:
        #         # 例えば2クラス分類モデルならスコアをsoftmax後に取るなど
        #         scores = logits[:, 1]

        # reranked = sorted(
        #     zip(unique_initial_docs, scores.tolist()), key = lambda x: x[1], reverse=True
        # )
        # result_docs = [doc for doc, score in reranked[:self.top_k]]

        # 一括処理がメモリ不足になる場合はバッチ処理
        batch_size = 8
        pairs = [(query, doc.page_content) for doc in unique_initial_docs]
        scores = []
        doc_refs = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                batch_docs = unique_initial_docs[i:i+batch_size]
                batch_inputs = self._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self._device)
                batch_logits = self._model(**batch_inputs).logits
                if batch_logits.ndim == 2 and batch_logits.shape[1] == 1:
                    batch_scores = batch_logits.squeeze(-1)
                else:
                    batch_scores = batch_logits[:, 1]
                scores.extend(batch_scores.tolist())
                doc_refs.extend(batch_docs)
        reranked = sorted(
            zip(doc_refs, scores), key=lambda x: x[1], reverse=True
        )
        result_docs = [doc for doc, score in reranked[:self.top_k]]
        print(f"再ランキング時間: {time.perf_counter() - start_rerank:.2f}秒\n\n")  
        return result_docs
    
    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return await asyncio.to_thread(self.get_relevant_documents, query, **kwargs)
    

# ruri v3 sentence transformer専用embeddingラッパー
class RuriEmbeddingWithPrefixV3: 
    def __init__(self, device = "cuda", dtype = torch.float16):
        self.model = SentenceTransformer("cl-nagoya/ruri-v3-310m", device=device)
        self.model = self.model.to(dtype)

    def embed_query(self, text):
        # 先頭に"検索クエリ: "を付けてからembeddingを計算(v3)
        return self.model.encode("検索クエリ: " + text, convert_to_numpy=True).tolist()

    def embed_documents(self, texts):
        # 各テキストに"検索文書: "を付けてからembeddingを計算
        return self.model.encode(
            ["検索文書: " + text for text in texts], 
            convert_to_numpy=True
        ).tolist()
    

# ruri v2専用embeddingラッパー
class RuriEmbeddingWithPrefix: 
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_query(self, text):
        # 先頭に"文章:"を付けてからembeddingを計算(v2)
        # v3は、"検索文書: "
        prefixed_text = "クエリ:" + text
        return self.embedding.embed_query(prefixed_text)

    def embed_documents(self, texts):
        # 各テキストに"ruri:"を付けてからembeddingを計算
        prefixed_texts = ["文章:" + text for text in texts]
        return self.embedding.embed_documents(prefixed_texts)

# キャッシュファイルを介さずに直接Chromaに登録する関数
def register_to_chroma_batch(
        docs, 
        embedding, 
        persist_directory="chroma_db", 
        batch_size=1000
):
    """
    docsをembeddingしてChromaに登録する。
    - docs: 登録するドキュメントのリスト
    - embedding: 使用する埋め込みモデル
    - persist_directory: Chromaの永続化ディレクトリ
    """
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=None
    )
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        # embeddingをバッチ分だけ生成
        batch_vecs = [embedding.embed_query(doc.page_content) for doc in batch_docs]
        db._collection.add(
            embeddings=batch_vecs,
            documents=[doc.page_content for doc in batch_docs],
            metadatas=[doc.metadata for doc in batch_docs],
            ids=[str(i + j) for j in range(len(batch_docs))]  # IDを文字列に変換
        )
    
        print(f"{i + len(batch_docs)}件目まで登録完了")

# embeddingキャッシュを使ってchromaに登録
def register_to_chroma_with_cache(
        docs, 
        embedding, 
        persist_directory="chroma_db", 
        cache_path="embeddings.pkl",
        batch_size=1500
):
    """
    docsをembeddingしてChromaに登録する。
    - docs: 登録するドキュメントのリスト
    - embedding: 使用する埋め込みモデル
    - persist_directory: Chromaの永続化ディレクトリ
    - cache_path: 埋め込みキャッシュファイルのパス
    """
    # キャッシュから埋め込みを取得または生成
    vecs = get_or_create_embeddings(docs, embedding, cache_path)
    if len(vecs) != len(docs):
        raise ValueError(f"埋め込みの数がドキュメントの数と一致しません: {len(vecs)} != {len(docs)}")

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=None
    )
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_vecs = vecs[i:i + batch_size]
        # ベクトルとドキュメントを一括で追加
        db._collection.add(
            embeddings=batch_vecs,
            documents=[doc.page_content for doc in batch_docs],
            metadatas=[doc.metadata for doc in batch_docs],
            ids=[str(i + j) for j in range(len(batch_docs))]  # IDを文字列に変換
        )
        db.persist()
        print(f"{i + len(batch_docs)}件目まで登録完了")

# vectordbキャッシュ
def get_or_create_embeddings(docs, embedding, cache_path="embeddings.pkl"):
    # 既存キャッシュがあれば読み込む
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            vecs = pickle.load(f)
    else:
        vecs = []

    start = len(vecs)
    print(f"キャッシュ済み: {start}件 / 全{len(docs)}件")
    for i, doc in enumerate(docs[start:], start):
        vecs.append(embedding.embed_query(doc.page_content))
        if i % 100 == 0:
            print(f"{i}件目までembedding生成")
            # 途中経過を保存
            with open(cache_path, "wb") as f:
                pickle.dump(vecs, f)
        # time.sleep(1)  # サーバー負荷軽減

    # 最終保存
    with open(cache_path, "wb") as f:
        pickle.dump(vecs, f)
    return vecs

# prefilter
def prefilter(keyw, docs): 
    prior_docs = []
    other_docs = []
    for doc in docs: 
        if any(kw in doc.page_content.lower() for kw in keyw) :
            prior_docs.append(doc)
        else:
            other_docs.append(doc)

    return prior_docs, other_docs

# retry付きadd_documents。未使用
def add_documents_with_retry(vectordb, batch, max_retries=5, wait_seconds=5):
    for attempt in range(1, max_retries + 1):
        try:
            vectordb.add_documents(documents=batch)
            return True
        except Exception as e:
            print(f"add_documents失敗（{attempt}回目）: {e}")
            if attempt < max_retries:
                print(f"{wait_seconds}秒待機してリトライします...")
                time.sleep(wait_seconds)
            else:
                print("リトライ回数上限に達しました。スキップします。")
                return False

# contextを作るときに、timestampを文頭に追加
def format_context(docs):
    prior_docs, other_docs = prefilter(["#continue"], docs) # #continueタグ付きドキュメントを優先
    all_docs = prior_docs + other_docs
    return "\n\n".join(
        f"[日時: {doc.metadata.get('timestamp', '不明')}] {doc.page_content}"
        for doc in all_docs
    )

# llmからの回答の形式に合わせて内容を取り出す
def extract_response_text(response):
    if hasattr(response, "content"):
        return response.content
    # dict型（resultキーあり）
    if isinstance(response, dict) and "result" in response:
        return response["result"]
    # str型
    if isinstance(response, str):
        return response
    # その他（例: listや他の型）
    return str(response)

def save_chat_history_markdown(chat_history, vaultdir, llm_name=None, filename=None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{vaultdir}/ai_generated/chat_history_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        if llm_name:
            f.write(f"LLM: {llm_name}    #ai_generated \n\n")
        for i, (q, a) in enumerate(chat_history, 1):
            f.write(f"### Q{i}\n{q}\n\n")
            f.write(f"**A{i}**\n{a}\n\n")

def remove_semantic_duplicates_fast(docs, embedding, threshold=0.95):
    contents = [doc.page_content for doc in docs]
    vecs = torch.tensor(embedding.embed_documents(contents)).to('cuda')
    keep = []
    mask = torch.ones(len(vecs), dtype=torch.bool, device=vecs.device)
    for i in range(len(vecs)):
        if not mask[i]:
            continue
        sims = torch.nn.functional.cosine_similarity(vecs[i].unsqueeze(0), vecs, dim=1)
        dup_idx = (sims >= threshold) & mask
        mask[dup_idx] = False
        mask[i] = True
        keep.append(docs[i])
    return keep

def remove_semantic_duplicates(docs, embedding, threshold=0.9):
    """
    docs: List[Document]
    embedding: 埋め込みモデル（embed_queryメソッドを持つ）
    threshold: 類似度がこの値以上なら重複とみなす
    """
    unique_docs = []
    embeddings = []
    for doc in docs:
        vec = np.array(embedding.embed_query(doc.page_content))
        is_duplicate = False
        for e in embeddings:
            sim = np.dot(vec, e) / (np.linalg.norm(vec) * np.linalg.norm(e))
            if sim >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_docs.append(doc)
            embeddings.append(vec)
    return unique_docs

def remove_think_tags(text):
    # <think> ... </think> をすべて除去
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def prefilter_by_timestamp(docs, after_datetime=None, before_datetime=None):
    """
    after_datetime: datetime 型。これ以降のtimestampを持つドキュメントをprior_docsとする
    before_datetime: datetime 型。これより前のtimestampを持つドキュメントをprior_docsとする
    """
    prior_docs = []
    other_docs = []
    for doc in docs:
        ts = doc.metadata.get("timestamp")
        if ts is not None:
            try:
                doc_time = datetime.datetime.fromisoformat(ts)
            except Exception:
                doc_time = None
        else:
            doc_time = None

        is_prior = False
        if doc_time is not None:
            if after_datetime and doc_time >= after_datetime:
                is_prior = True
            if before_datetime and doc_time <= before_datetime:
                is_prior = True
        if is_prior:
            prior_docs.append(doc)
        else:
            other_docs.append(doc)
    return prior_docs, other_docs

import datetime

def summarize_recent_docs(docs, llm, days=7):
    """
    docs: List[Document]
    llm: LLMインスタンス（.invoke(prompt)で要約できるもの）
    直近の指定された日数のドキュメントを抽出し、LLMでまとめた要約文を返す
    """
    print(f"直近{days}日間のドキュメントを要約します...")
    start_recent = time.perf_counter()
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=days)
    # 直近の指定された日数のドキュメントを抽出
    recent_docs, _ = prefilter_by_timestamp(docs, after_datetime=start_date)
    if not recent_docs:
        return f"直近{days}日間のドキュメントはありません。"

    # まとめるテキストを作成
    context = "\n\n".join(
        f"[日時: {doc.metadata.get('timestamp', '不明')}] {doc.page_content}"
        for doc in recent_docs
    )
    prompt = (
        "以下は直近のドキュメントです。内容を日本語で要約してください。\n\n"
        f"{context}\n\n要約:"
    )
    summary = llm.invoke(prompt)
    summary = extract_response_text(summary)
    summary = remove_think_tags(summary)
    end_recent = time.perf_counter()
    print(f"直近{days}日間のドキュメント数: {len(recent_docs)}、要約時間: {end_recent - start_recent:.2f}秒")   
    print(f"=== 直近{days}日間の要約 ===\n{summary}\n========================")
    return summary

def random_model(a_models_list):
    model = np.random.choice(a_models_list)
    print(f"Randomly selected model: {model}")
    if model == "gpt-4.1":
        llm = AzureAIChatCompletionsModel(
            endpoint = os.getenv("AZURE_GPT41_ENDPOINT"),
            credential = os.getenv("AZURE_GPT41_CREDENTIAL"),
            model = "gpt-4.1"
        )
    elif model == "gpt-5.2-chat":
        llm = AzureAIChatCompletionsModel(
            endpoint = os.getenv("AZURE_GPT52_ENDPOINT"),
            credential = os.getenv("AZURE_GPT52_CREDENTIAL"),
            model = "gpt-5.2-chat"
        )
    else:
        llm = OllamaLLM(model = model)
    return llm

if __name__ == "__main__":
    main()
