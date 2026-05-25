#!/usr/bin/env python3
"""第八章环境连通性自检：LLM / Embedding / Qdrant / Neo4j / spaCy"""

from dotenv import load_dotenv

load_dotenv(override=True)

import os
import sys


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def fail(msg: str) -> None:
    print(f"  ❌ {msg}")


def check_env() -> bool:
    print("\n[1/6] 环境变量")
    required = {
        "LLM_MODEL_ID": os.getenv("LLM_MODEL_ID"),
        "LLM_API_KEY": os.getenv("LLM_API_KEY"),
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL"),
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
        "EMBED_API_KEY": os.getenv("EMBED_API_KEY"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        fail(f"缺少: {', '.join(missing)}")
        return False
    ok(f"LLM={required['LLM_MODEL_ID']}, Embed={os.getenv('EMBED_MODEL_TYPE', 'dashscope')}")
    return True


def check_spacy() -> bool:
    print("\n[2/6] spaCy 语言模型")
    passed = True
    for model in ("zh_core_web_sm", "en_core_web_sm"):
        try:
            import spacy

            spacy.load(model)
            ok(model)
        except Exception as e:
            fail(f"{model}: {e}")
            passed = False
    return passed


def check_embedding() -> bool:
    print("\n[3/6] Embedding (DashScope)")
    try:
        from hello_agents.memory.embedding import get_text_embedder, get_dimension

        emb = get_text_embedder()
        vec = emb.encode("HelloAgents 连通性测试")
        dim = get_dimension()
        ok(f"向量维度 {len(vec)}（配置 QDRANT_VECTOR_SIZE={os.getenv('QDRANT_VECTOR_SIZE', '未设置')}）")
        if os.getenv("QDRANT_VECTOR_SIZE") and int(os.getenv("QDRANT_VECTOR_SIZE")) != dim:
            fail(f"QDRANT_VECTOR_SIZE 与 Embedding 维度不一致，建议改为 {dim}")
            return False
        return True
    except Exception as e:
        fail(str(e))
        return False


def check_qdrant() -> bool:
    print("\n[4/6] Qdrant")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "30")),
        )
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        ok(f"连接成功，已有集合 {len(names)} 个")
        return True
    except Exception as e:
        fail(str(e))
        return False


def check_neo4j() -> bool:
    print("\n[5/6] Neo4j")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            record = session.run("RETURN 1 AS n").single()
            assert record["n"] == 1
        driver.close()
        ok("连接成功")
        return True
    except Exception as e:
        fail(str(e))
        return False


def check_llm() -> bool:
    print("\n[6/6] LLM (DeepSeek)")
    try:
        from hello_agents import HelloAgentsLLM

        llm = HelloAgentsLLM()
        reply = llm.invoke([{"role": "user", "content": "只回复两个字：通过"}])
        ok(f"响应: {reply.strip()[:50]}")
        return True
    except Exception as e:
        fail(str(e))
        return False


def main() -> None:
    print("=" * 50)
    print("HelloAgents Chapter8 连通性自检")
    print("=" * 50)

    results = [
        check_env(),
        check_spacy(),
        check_embedding(),
        check_qdrant(),
        check_neo4j(),
        check_llm(),
    ]

    print("\n" + "=" * 50)
    if all(results):
        print("🎉 全部通过，可以运行 chapter8 示例")
        sys.exit(0)
    print(f"⚠️  {sum(results)}/{len(results)} 项通过，请根据上方 ❌ 排查")
    sys.exit(1)


if __name__ == "__main__":
    main()
