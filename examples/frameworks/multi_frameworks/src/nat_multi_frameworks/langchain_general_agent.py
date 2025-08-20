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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LangChainGeneralAgentConfig(FunctionBaseConfig, name="langchain_general_agent"):
    llm_name: LLMRef


@register_function(config_type=LangChainGeneralAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def langchain_general_agent_as_tool(tool_config: LangChainGeneralAgentConfig, builder: Builder):

    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Define a simple prompt for general chatbot responses
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Answer the following question in a friendly and informative way:\n\n{question}"
    )

    # Get the LLM from the builder
    llm = await builder.get_llm(llm_name=tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Create a simple chain for general questions
    chain = prompt | llm | StrOutputParser()

    async def _arun(inputs: str) -> str:
        """
        Handle general questions using LangChain without RAG.
        Args:
            inputs: User input
        """
        try:
            output = await chain.ainvoke({"question": inputs})
            logger.info("Output from LangChain general agent: %s", output)
            return output
        except Exception as e:
            logger.error("Error in LangChain general agent: %s", e)
            return "Sorry, I couldn't process your request at the moment."

    yield FunctionInfo.from_fn(
        _arun,
        description="Answer general questions and chitchat using LangChain without RAG",
    )
