#!/usr/bin/env python3
# ruff: noqa: T201
"""
Example CrewAI integration with MemMachine.

This example demonstrates how to use MemMachine memory tools with CrewAI agents.
"""

import os

from crewai import Agent, Crew, Task
from integrations.crewai.tool import create_memmachine_tools

# Configuration
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
ORG_ID = os.getenv("CREWAI_ORG_ID", "example_org")
PROJECT_ID = os.getenv("CREWAI_PROJECT_ID", "example_project")
USER_ID = os.getenv("CREWAI_USER_ID", "user_001")

# LLM: use OpenAI (from OPENAI_API_KEY in .env) unless Qwen is explicitly set
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
if QWEN_API_KEY:
    os.environ["OPENAI_API_KEY"] = QWEN_API_KEY
    os.environ["OPENAI_BASE_URL"] = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwen_model = os.getenv("QWEN_MODEL", "qwen-turbo")
    os.environ["OPENAI_MODEL_NAME"] = qwen_model
    os.environ["OPENAI_MODEL"] = qwen_model
# else: CrewAI uses OPENAI_API_KEY and OPENAI_MODEL from your environment (e.g. .env)


def main() -> None:
    """Run a simple CrewAI example with MemMachine memory."""
    print("=" * 60)
    print("CrewAI + MemMachine Integration Example")
    print("=" * 60)
    print()

    # Create MemMachine tools
    print("Creating MemMachine tools...")
    memmachine_tools = create_memmachine_tools(
        base_url=MEMORY_BACKEND_URL,
        org_id=ORG_ID,
        project_id=PROJECT_ID,
        user_id=USER_ID,
    )
    print(f"✓ Created {len(memmachine_tools)} tools")
    print()

    # Create research agent
    print("Creating research agent...")
    researcher = Agent(
        role="Research Assistant",
        goal="Research topics thoroughly and remember key findings for future reference",
        backstory="""You are an expert researcher with excellent memory capabilities.
        You always search memory first to see if you've researched a topic before.
        When you find new information, you store it in memory so you can recall it later.
        Store only substantive research findings and summaries in memory—never store meta-commentary
        like "No prior memory found" or "Proceeding with fresh research."
        You provide comprehensive, well-researched summaries.""",
        tools=memmachine_tools,
        verbose=True,
        allow_delegation=False,
    )
    print("✓ Research agent created")
    print()

    # Create writing agent (shares same memory tools so it can reference research)
    print("Creating writing agent...")
    writer = Agent(
        role="Writing Assistant",
        goal="Write clear, well-structured articles that cite research from memory",
        backstory="""You are an experienced writer who produces articles and reports.
        You always search memory first to find relevant research and data to reference.
        You cite and build on that research in your writing rather than making unsupported claims.
        You write in a clear, engaging style and structure articles with a brief intro, main points, and conclusion.""",
        tools=memmachine_tools,
        verbose=True,
        allow_delegation=False,
    )
    print("✓ Writing agent created")
    print()

    # Create research tasks
    print("Creating research tasks...")
    task_research_healthcare = Task(
        description="""Research the topic: {topic}

        Steps:
        1. First, search your memory to see if you've researched this topic before.
        2. If you find previous research, do the following:
        - Extract 1–2 *relevant* takeaways from memory (paraphrase; do not quote more than ~15 words).
        - Include them at the top under a short header: "Context from memory (brief)".
        - Then build upon them naturally in your research summary.
        3. Research the topic thoroughly using your available tools.
        4. Store only key findings and research summaries in memory (do not store meta-commentary such as
        "No prior memory found" or "Proceeding with fresh research").
        5. Provide a comprehensive summary of your research.

        Output format:
        - If memory was found, start with:
        Context from memory (brief):
        - <takeaway 1>
        - <takeaway 2 (optional)>
        - Then provide:
        Research summary:
        <your summary>

        Remember to use the search_memory tool first, then add_memory to store only substantive findings.""",
        agent=researcher,
        expected_output="A comprehensive research summary with key findings stored in memory",
    )

    task_research_diagnosis = Task(
        description="""Research the topic: AI applications in medical diagnosis.

        Steps:
        1. First, search your memory to see if you've researched this topic before.
        2. If you find previous research, extract 1–2 relevant takeaways and include them under "Context from memory (brief)" at the top, then build on them.
        3. Research the topic thoroughly using your available tools.
        4. Store only key findings and research summaries in memory (no meta-commentary).
        5. Provide a comprehensive summary.

        Output format: If memory was found, start with Context from memory (brief), then Research summary.""",
        agent=researcher,
        expected_output="A comprehensive research summary with key findings stored in memory",
        context=[task_research_healthcare],
    )

    task_article = Task(
        description="""Write an article about AI in healthcare (roughly 400–600 words) that references the research stored in memory.

        Steps:
        1. Search memory for research on "AI in healthcare" and "medical diagnosis" (or related terms) to find the research your teammate has stored.
        2. Use that research as your main source: cite specific points, applications, and findings from the search results.
        3. Write a well-structured article with:
           - A short engaging introduction on AI in healthcare
           - Main sections (e.g. applications, benefits, challenges) that explicitly reference the research from memory
           - A brief conclusion
        4. Do not invent statistics or studies—base the article on what you find in memory. If memory has little content, say so and write a short note instead of fabricating.""",
        agent=writer,
        expected_output="An article about AI in healthcare that cites research from memory, with intro, main sections, and conclusion",
        context=[task_research_healthcare, task_research_diagnosis],
    )
    print("✓ Tasks created")
    print()

    # Create and run the crew (research tasks run first, then writer uses memory)
    print("Creating crew...")
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task_research_healthcare, task_research_diagnosis, task_article],
        verbose=True,
    )
    print("✓ Crew created")
    print()

    # Run the crew: researcher runs twice (two topics), then writer writes article from memory
    print("Running crew (research → research → article)...")
    print("-" * 60)
    result = crew.kickoff(inputs={"topic": "Artificial Intelligence in Healthcare"})
    print("-" * 60)
    print()

    print("=" * 60)
    print("Final result (article referencing research from memory):")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
