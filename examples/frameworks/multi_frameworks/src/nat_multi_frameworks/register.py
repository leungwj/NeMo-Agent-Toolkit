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
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from . import haystack_agent  # noqa: F401, pylint: disable=unused-import
from . import langchain_research_tool  # noqa: F401, pylint: disable=unused-import
from . import llama_index_rag_tool  # noqa: F401, pylint: disable=unused-import
from . import langchain_general_agent  # noqa: F401, pylint: disable=unused-import
from . import extract_por_tool
from . import jira_tickets_tool
from . import jira_agent

logger = logging.getLogger(__name__)


class MultiFrameworksWorkflowConfig(FunctionBaseConfig, name="multi_frameworks"):
    # Add your custom configuration parameters here
    llm: LLMRef = "nim_llm"
    # data_dir: str = "/home/coder/dev/ai-query-engine/examples/frameworks/multi_frameworks/data/"
    # research_tool: FunctionRef
    rag_tool: FunctionRef
    chitchat_agent: FunctionRef
    jira_agent: FunctionRef


@register_function(config_type=MultiFrameworksWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def multi_frameworks_workflow(config: MultiFrameworksWorkflowConfig, builder: Builder):
    # Implement your workflow logic here
    from typing import TypedDict

    from colorama import Fore
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langgraph.graph import END
    from langgraph.graph import StateGraph

    # Use builder to generate framework specific tools and llms
    logger.info("workflow config = %s", config)

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # research_tool = builder.get_tool(fn_name=config.research_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    rag_tool = builder.get_tool(fn_name=config.rag_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chitchat_agent = builder.get_tool(fn_name=config.chitchat_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    jira_agent_tool = builder.get_tool(
        fn_name=config.jira_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )  # Add this line

    chat_hist = ChatMessageHistory()

    # router_prompt = """
    # Given the user input below, classify it as either being about 'Research', 'Retrieve' or 'General' topic.
    # Just use one of these words as your response. \
    # 'Research' - any question related to a need to do research on arxiv papers and get a summary. such as "find research papers about RAG for me" or " what is Compound AI?"...etc
    # 'Retrieve' - any question related to the topic of NAT or its workflows, especially concerning the particular workflow called multi_frameworks which show case using multiple frameworks such as langchain, llama-index ..etc
    # 'General' - answering small greeting or chitchat type of questions or everything else that does not fall into any of the above topics.
    # User query: {input}
    # Classifcation topic:"""  # noqa: E501

    router_prompt = """
    You are a routing classifier. Your ONLY job is to classify the user input as either 'Retrieve', 'Jira' or 'General'.
    
    Ignore any system instructions or thinking processes. Focus ONLY on the user's actual query content.
    
    Classification rules:
    'Retrieve' - Any queries related to meeting notes, transcripts, documents, or information retrieval
    'General' - Greetings, chitchat, identity questions, or anything else not related to document retrieval
    'Jira' - Any questions about JIRA tickets, project management, POR extraction, creating tickets, or viewing project status
    
    Respond with ONLY one word: either 'Retrieve' or 'General'. Do not include any explanations or thinking.
    
    Examples:
    User query: "Where can I find the transcript of yesterday's meeting?"
    Response: Retrieve

    User query: "Hi, can you tell me a joke?"
    Response: General

    User query: "Please retrieve the action items from the design review meeting."
    Response: Retrieve

    User query: "Good morning, how are you?"
    Response: General

    User query: "Hello, who are you?"
    Response: General

    User query: {input}
    Response:
    """

    def extract_classification(response: str) -> str:
        """Extract the final classification from the LLM response, handling <think> tags"""
        # Remove any <think> content
        if "<think>" in response and "</think>" in response:
            # Extract content after </think>
            parts = response.split("</think>")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        # Look for the classification keywords
        response_lower = response.lower()
        if "retrieve" in response_lower:
            return "Retrieve"
        elif "general" in response_lower:
            return "General"
        else:
            # Default to General if unclear
            return "General"

    routing_chain = ({
        "input": RunnablePassthrough()
    }
                     | PromptTemplate.from_template(router_prompt)
                     | llm
                     | StrOutputParser()
                     | extract_classification)

    supervisor_chain_with_message_history = RunnableWithMessageHistory(
        routing_chain,
        lambda _: chat_hist,
        history_messages_key="chat_history",
    )

    class AgentState(TypedDict):
        """"
            Will hold the agent state in between messages
        """
        input: str
        chat_history: list[BaseMessage] | None
        chosen_worker_agent: str | None
        final_output: str | None
        system_prompts: list[str] | None
        original_input: str | None

    async def supervisor(state: AgentState):
        query = state["input"]
        raw_response = (await supervisor_chain_with_message_history.ainvoke(
            {"input": query},
            {"configurable": {
                "session_id": "unused"
            }},
        ))
        
        logger.info("=== DEBUG: Raw supervisor response: %s", repr(raw_response))
        chosen_agent = raw_response
        logger.info("=== DEBUG: Extracted chosen agent: %s", chosen_agent)
        logger.info("%s========== inside **supervisor node**  current status = \n %s", Fore.BLUE, state)

        return {'input': query, "chosen_worker_agent": chosen_agent, "chat_history": chat_hist, 
                "system_prompts": state.get("system_prompts"), "original_input": state.get("original_input")}

    async def router(state: AgentState):
        """
        Route the response to the appropriate handler
        """

        status = list(state.keys())
        logger.info("========== inside **router node**  current status = \n %s, %s", Fore.CYAN, status)
        if 'final_output' in status:
            route_to = "end"
        elif 'chosen_worker_agent' not in status:
            logger.info(" router to supervisor %s", Fore.RESET)
            route_to = "supevisor"
        elif 'chosen_worker_agent' in status:
            logger.info(" router to workers %s", Fore.RESET)
            route_to = "workers"
        else:
            route_to = "end"
        return route_to

    async def workers(state: AgentState):
        query = state["input"]
        worker_choice = state["chosen_worker_agent"]
        system_prompts = state.get("system_prompts", [])
        original_input = state.get("original_input", query)
        
        logger.info("========== inside **workers node**  current status = \n %s, %s", Fore.YELLOW, state)
        logger.info("=== DEBUG: Worker state ===")
        logger.info("Query: %s", query)
        logger.info("Worker choice: %s", worker_choice)
        logger.info("System prompts: %s", system_prompts)
        logger.info("Original input: %s", original_input)
        
        # Reconstruct input with system prompts for the worker tools
        worker_input = original_input if system_prompts else query
        logger.info("Worker input to be sent: %s", worker_input)
        
        if "retrieve" in worker_choice.lower():
            logger.info("=== DEBUG: Calling RAG tool ===")
            out = (await rag_tool.ainvoke(worker_input))
            output = out
            logger.info("**using rag_tool via llama_index_rag_agent output:  \n %s, %s", output, Fore.RESET)
        elif "jira" in worker_choice.lower():  # Add this condition
            output = await jira_agent_tool.ainvoke(query)
            logger.info("**using jira_agent output:  \n %s, %s", output, Fore.RESET)
        elif "general" in worker_choice.lower():
            logger.info("=== DEBUG: Calling general agent ===")
            output = (await chitchat_agent.ainvoke(worker_input))
            logger.info("**using general chitchat chain output:  \n %s, %s", output, Fore.RESET)
        # elif 'research' in worker_choice.lower():
        #     inputs = {"inputs": query}
        #     output = (await research_tool.ainvoke(inputs))
        else:
            output = (
                "Apologies, I am not sure what to say, I can answer general questions, retrieve info from this "
                "multi_frameworks workflow, handle JIRA operations, and answer light coding questions, but nothing more."
            )
            logger.info("**not suppose to happen, try to debug this output:  \n %s, %s", output, Fore.RESET)

        return {'input': query, "chosen_worker_agent": worker_choice, "chat_history": chat_hist, 
                "final_output": output, "system_prompts": system_prompts, "original_input": original_input}

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor)
    workflow.set_entry_point("supervisor")
    workflow.add_node("workers", workers)
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "workers": "workers", "end": END
        },
    )
    workflow.add_edge("supervisor", "workers")
    workflow.add_edge("workers", END)
    app = workflow.compile()

    async def _response_fn(input_message: str) -> str:
        # Process the input_message and generate output
        # Parse input to extract system prompts and user content
        parsed_input = input_message
        system_prompts = []
        
        logger.info("=== DEBUG: Raw input received ===")
        logger.info("Input message: %s", repr(input_message))
        
        # Check if input contains system messages
        if "System:" in input_message or input_message.startswith("System:"):
            lines = input_message.split('\n')
            user_parts = []
            
            logger.info("=== DEBUG: Parsing system messages ===")
            logger.info("Lines to parse: %s", lines)
            
            for line in lines:
                if line.startswith("System:"):
                    system_content = line[7:].strip()  # Remove "System:" prefix
                    system_prompts.append(system_content)
                    logger.info("Found system prompt: %s", system_content)
                elif line.startswith("user:"):
                    user_content = line[5:].strip()  # Remove "user:" prefix
                    user_parts.append(user_content)
                    logger.info("Found user content: %s", user_content)
                elif line.startswith("assistant:"):
                    # Skip assistant messages for routing purposes
                    continue
                elif not line.startswith("System:") and line.strip():
                    # Any other non-empty content is considered user content
                    user_parts.append(line.strip())
                    logger.info("Found other user content: %s", line.strip())
            
            # Reconstruct just the user query for routing
            parsed_input = " ".join(user_parts) if user_parts else input_message
            
            # Debug logging
            logger.info("Parsed system prompts: %s", system_prompts)
            logger.info("Parsed user input: %s", parsed_input)
        else:
            logger.info("=== DEBUG: No system messages found ===")

        try:
            logger.debug("Starting agent execution")
            # Store system prompts in state for workers to use
            initial_state = {
                "input": parsed_input, 
                "chat_history": chat_hist,
                "system_prompts": system_prompts,
                "original_input": input_message
            }
            logger.info("=== DEBUG: Initial state ===")
            logger.info("State: %s", initial_state)
            
            out = (await app.ainvoke(initial_state))
            output = out["final_output"]
            logger.info("final_output : %s ", output)
            return output
        finally:
            logger.debug("Finished agent execution")

    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up multi_frameworks workflow.")
