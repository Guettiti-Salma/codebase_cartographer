"""
graph.py
--------
Wires the node functions from nodes.py into an actual LangGraph
StateGraph: which node runs first, which nodes loop back on themselves,
and which nodes branch based on a condition function.
"""

from langgraph.graph import StateGraph, START, END          # graph builder + the fixed start/end markers
from langgraph.checkpoint.memory import MemorySaver          # in-memory state persistence between steps

from .state import RepoState                                 # the state shape every node shares
from . import nodes as n                                     # all node functions, namespaced as `n.<name>`


def build_graph():
    graph = StateGraph(RepoState)                             # a graph whose shared state follows the RepoState shape

    # --- register every node under a string name (used to wire edges below) ---
    graph.add_node("clone_repo", n.clone_repo_node)
    graph.add_node("chunk_and_index", n.chunk_and_index_node)
    graph.add_node("identify_entry_points", n.identify_entry_points_node)
    graph.add_node("trace_step", n.trace_step_node)
    graph.add_node("summarize_modules", n.summarize_modules_node)
    graph.add_node("generate_diagram", n.generate_diagram_node)
    graph.add_node("validate_diagram", n.validate_diagram_node)
    graph.add_node("finalize_diagram", n.finalize_diagram_node)
    graph.add_node("generate_writeup", n.generate_writeup_node)

    # --- the straight-line part of the pipeline: setup steps, run once each ---
    graph.add_edge(START, "clone_repo")
    graph.add_edge("clone_repo", "chunk_and_index")
    graph.add_edge("chunk_and_index", "identify_entry_points")
    graph.add_edge("identify_entry_points", "trace_step")

    # --- loop #1: keep tracing imports until the frontier queue is empty ---
    graph.add_conditional_edges(
        "trace_step",
        n.frontier_router,                                    # returns the string "loop" or "done"
        {"loop": "trace_step", "done": "summarize_modules"},   # map those strings to actual node names
    )

    graph.add_edge("summarize_modules", "generate_diagram")
    graph.add_edge("generate_diagram", "validate_diagram")

    # --- loop #2: keep retrying the diagram until it's valid or we hit the retry cap ---
    graph.add_conditional_edges(
        "validate_diagram",
        n.diagram_router,                                      # returns "retry" or "proceed"
        {"retry": "generate_diagram", "proceed": "finalize_diagram"},
    )

    graph.add_edge("finalize_diagram", "generate_writeup")
    graph.add_edge("generate_writeup", END)

    # a checkpointer lets a run's state be saved after every node — not
    # strictly required for a one-shot CLI script, but it's what would let
    # this same graph power a resumable/interactive version later (e.g. a
    # web UI) without restructuring anything above this line
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
