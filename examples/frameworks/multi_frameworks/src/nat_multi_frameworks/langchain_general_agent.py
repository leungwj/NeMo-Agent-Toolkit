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
import re

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
            inputs: User input (may include system messages in format "System: <msg>\nuser: <msg>")
        """
        try:
            # Parse the input to extract messages and reconstruct proper message format
            if inputs.startswith("System:") or "System:" in inputs:
                # Parse the structured input to extract system and user messages
                lines = inputs.split('\n')
                messages = []
                
                for line in lines:
                    if line.startswith("System:"):
                        system_content = line[7:].strip()  # Remove "System:" prefix
                        messages.append({"role": "system", "content": system_content})
                        logger.info("Found system message: %s", system_content)
                    elif line.startswith("user:"):
                        user_content = line[5:].strip()  # Remove "user:" prefix
                        messages.append({"role": "user", "content": user_content})
                    elif line.startswith("assistant:"):
                        assistant_content = line[10:].strip()  # Remove "assistant:" prefix
                        messages.append({"role": "assistant", "content": assistant_content})
                    elif not line.startswith("System:") and line.strip():
                        # Any other non-empty content is considered user content
                        messages.append({"role": "user", "content": line.strip()})
                
                # Convert to LangChain message format
                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
                
                langchain_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        langchain_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                logger.info("Converted to %d LangChain messages", len(langchain_messages))
                # Call LLM with structured messages
                output = await llm.ainvoke(langchain_messages)
                
            else:
                # Fallback for unstructured input - use the old approach
                if "System:" in inputs or inputs.startswith("System:"):
                    # Even unstructured input might have system prompts, let's try to parse them
                    lines = inputs.split('\n')
                    messages = []
                    
                    for line in lines:
                        if line.startswith("System:"):
                            system_content = line[7:].strip()  # Remove "System:" prefix
                            messages.append({"role": "system", "content": system_content})
                            logger.info("Found system message in fallback: %s", system_content)
                        elif line.startswith("user:"):
                            user_content = line[5:].strip()  # Remove "user:" prefix
                            messages.append({"role": "user", "content": user_content})
                        elif not line.startswith("System:") and not line.startswith("assistant:"):
                            # Any other content is considered user content
                            if line.strip():
                                messages.append({"role": "user", "content": line.strip()})
                    
                    if messages:
                        # Convert to LangChain message format
                        from langchain_core.messages import SystemMessage, HumanMessage
                        
                        langchain_messages = []
                        for msg in messages:
                            if msg["role"] == "system":
                                langchain_messages.append(SystemMessage(content=msg["content"]))
                            elif msg["role"] == "user":
                                langchain_messages.append(HumanMessage(content=msg["content"]))
                        
                        # Call LLM with structured messages
                        output = await llm.ainvoke(langchain_messages)
                    else:
                        # Fallback to chain
                        formatted_input = "You are a helpful assistant. Answer the following question in a friendly and informative way:\n\n" + inputs
                        output = await chain.ainvoke({"question": inputs})
                else:
                    # Use the chain for pure user input
                    output = await chain.ainvoke({"question": inputs})
                
            # Post-process the output to handle /no_think
            final_output = output.content if hasattr(output, 'content') else str(output)
            
            # Check if no_think was requested and remove thinking sections
            if "/no_think" in inputs and "<think>" in final_output:
                final_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL)
                final_output = final_output.strip()
                
            logger.info("Output from LangChain general agent: %s", final_output)
            return final_output
            
        except Exception as e:
            logger.error("Error in LangChain general agent: %s", e)
            return "Sorry, I couldn't process your request at the moment."

    yield FunctionInfo.from_fn(
        _arun,
        description="Answer general questions and chitchat using LangChain without RAG",
    )
