import logging
import re
from typing import Literal

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef, LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class JiraAgentConfig(FunctionBaseConfig, name="jira_agent"):
    llm_name: LLMRef
    extract_por_tool: FunctionRef
    show_jira_tickets_tool: FunctionRef
    create_jira_tickets_tool: FunctionRef
    get_jira_tickets_tool: FunctionRef
    # hitl_approval_tool: FunctionRef = None  # Optional


@register_function(config_type=JiraAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def jira_agent(tool_config: JiraAgentConfig, builder: Builder):
    
    import os
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel, Field

    api_token = os.getenv("NVIDIA_API_KEY")
    os.environ["NVIDIA_API_KEY"] = api_token

    if not api_token:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    # Get LLM and tools
    llm = await builder.get_llm(llm_name=tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    extract_por = builder.get_tool(fn_name=tool_config.extract_por_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    show_tickets = builder.get_tool(fn_name=tool_config.show_jira_tickets_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    create_tickets = builder.get_tool(fn_name=tool_config.create_jira_tickets_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    get_tickets = builder.get_tool(fn_name=tool_config.get_jira_tickets_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Intent classification prompt (similar to topic extraction)
    intent_prompt = """
    You are an expert at analyzing user requests related to JIRA and project management.
    If the user wants to extract POR requirements, then the context would be the file name defined by the user.
    
    User Input:
    ------
    {inputs}
    ------
    
    Classify the intent and extract relevant information.
    """
    
    prompt = PromptTemplate(
        input_variables=['inputs'],
        template=intent_prompt,
    )

    class JiraIntent(BaseModel):
        action: Literal["extract_por", "show_tickets", "create_tickets", "get_tickets"] = Field(
            description="The JIRA action the user wants to perform"
        )
        context: str = Field(description="Additional context or data for the action")

    llm_with_structure = llm.with_structured_output(JiraIntent)

    async def execute_jira_action(intent_result):
        try:
            action = intent_result.action
            context = intent_result.context

            logger.info("Action is: %s", action)
            logger.info("Context is: %s", context)
            
            if action == "extract_por":
                result = await extract_por.ainvoke(context)
            elif action == "show_tickets":
                result = await show_tickets.ainvoke(context)
            elif action == "create_tickets":
                result = await create_tickets.ainvoke(context)
            elif action == "get_tickets":
                result = await get_tickets.ainvoke(context)
            else:
                result = f"Unknown action: {action}"
                
            return result
            
        except Exception as e:
            logger.exception("Error executing JIRA action: %s", e)
            return f"Error executing JIRA action {action}: {e}"

    # Chain: intent classification → tool execution
    jira_chain = (prompt | llm_with_structure | execute_jira_action)

    async def _arun(inputs: str) -> str:
        """
        Process JIRA-related requests by determining intent and calling appropriate tools
        Args:
            inputs: user input related to JIRA operations
        """
        try:
            output = await jira_chain.ainvoke({"inputs": inputs})
            logger.info("Output from jira_agent: %s", output)
            return output
        except Exception as e:
            logger.error("Error in JIRA agent: %s", e)
            return f"Error querying JIRA: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description="Handle JIRA ticket operations including POR extraction, ticket creation, and ticket management")