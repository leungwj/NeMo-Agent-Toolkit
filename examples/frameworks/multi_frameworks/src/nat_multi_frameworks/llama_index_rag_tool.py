# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

from pydantic import ConfigDict

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LlamaIndexRAGConfig(FunctionBaseConfig, name="llama_index_rag"):

    model_config = ConfigDict(protected_namespaces=())

    llm_name: LLMRef
    embedding_name: EmbedderRef
    # Replace data_dir with milvus configuration
    milvus_uri: str = "http://localhost:19530"
    collection_name: str = "test_documents"  # Default collection name
    api_key: str | None = None
    model_name: str
    # Optional: for similarity search configuration
    similarity_top_k: int = 2


@register_function(config_type=LlamaIndexRAGConfig, framework_wrappers=[LLMFrameworkEnum.LLAMA_INDEX])
async def llama_index_rag_tool(tool_config: LlamaIndexRAGConfig, builder: Builder):

    from colorama import Fore
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex
    from llama_index.core.agent import FunctionCallingAgentWorker
    from llama_index.core.node_parser import SimpleFileNodeParser
    from llama_index.core.tools import QueryEngineTool
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from llama_index.core.storage.storage_context import StorageContext

    if (not tool_config.api_key):
        tool_config.api_key = os.getenv("NVIDIA_API_KEY")

    if not tool_config.api_key:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    logger.info(
        "##### Connecting to Milvus at %s, collection: %s",
        tool_config.milvus_uri,
        tool_config.collection_name,
    )

    llm = await builder.get_llm(tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
    embedder = await builder.get_embedder(tool_config.embedding_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

    Settings.embed_model = embedder
    # md_docs = SimpleDirectoryReader(input_files=[tool_config.data_dir]).load_data()
    # parser = SimpleFileNodeParser()
    # nodes = parser.get_nodes_from_documents(md_docs)
    # index = VectorStoreIndex(nodes)
    Settings.llm = llm

    # Get embedding dimensions dynamically
    try:
        test_embedding = await embedder.aget_text_embedding("test")
        embedding_dim = len(test_embedding)
        logger.info("Detected embedding dimension: %d", embedding_dim)
    except Exception as e:
        logger.warning(
            "Could not detect embedding dimension: %s. Using default 1024", e
        )
        embedding_dim = 1024

    # Create Milvus vector store
    vector_store = MilvusVectorStore(
        uri=tool_config.milvus_uri,
        collection_name=tool_config.collection_name,
        dim=embedding_dim,
        overwrite=False,  # Don't overwrite existing collection
        consistency_level="Session",
    )

    # Create storage context with Milvus
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index from existing vector store (no need to load documents)
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
        logger.info(
            "Successfully connected to existing Milvus collection: %s",
            tool_config.collection_name,
        )
    except Exception as e:
        logger.error(
            "Failed to connect to Milvus collection %s: %s",
            tool_config.collection_name,
            e,
        )
        raise ValueError(
            f"Could not connect to Milvus collection '{tool_config.collection_name}'. "
            f"Make sure the collection exists and contains data. Error: {e}"
        )

    query_engine = index.as_query_engine(similarity_top_k=tool_config.similarity_top_k)

    model_name = tool_config.model_name
    if not model_name.startswith('nvdev'):
        tool = QueryEngineTool.from_defaults(
            query_engine,
            name="rag",
            description="Query data from Milvus vector store using RAG with llama_index",
        )

        agent_worker = FunctionCallingAgentWorker.from_tools(
            [tool],
            llm=llm,
            verbose=True,
        )
        agent = agent_worker.as_agent()

    async def _arun(inputs: str) -> str:
        """
        RAG using llama-index querying Milvus vector store
        Args:
            inputs: user query
        """
        try:
            if not model_name.startswith("nvdev"):
                agent_response = await agent.achat(inputs)
                logger.info(
                    "Response from llama-index Agent querying Milvus: \n %s %s",
                    Fore.MAGENTA,
                    agent_response.response,
                )
                output = agent_response.response
            else:
                logger.info(
                    "%s Querying Milvus directly: %s %s %s",
                    Fore.MAGENTA,
                    type(query_engine),
                    query_engine,
                    inputs,
                )
                response = query_engine.query(inputs)
                output = response.response

                # Log source information if available
                if hasattr(response, "source_nodes") and response.source_nodes:
                    logger.info(
                        "Sources found: %d documents", len(response.source_nodes)
                    )
                    for i, node in enumerate(
                        response.source_nodes[:3]
                    ):  # Log first 3 sources
                        source = node.metadata.get("source", "Unknown")
                        score = getattr(node, "score", "N/A")
                        logger.info("Source %d: %s (score: %s)", i + 1, source, score)

            return output

        except Exception as e:
            logger.error("Error during RAG query: %s", e)
            return f"Error querying the knowledge base: {str(e)}"

    yield FunctionInfo.from_fn(
        _arun,
        description="Query relevant data from Milvus vector store via llama-index RAG",
    )
