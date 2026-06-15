import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from trajectory import Trajectory, Node


def visualize_tree_structure(root_node: Node, gt_node: Node, trajectory: Trajectory, output_path: str):
    """
    Visualize the hierarchical tree structure with trajectory actions.

    Args:
        root_node: The root node of the tree
        gt_node: The ground truth node for reference
        trajectory: The trajectory containing actions and turns
        output_path: Path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(18, 12))

    # Find the node that generated the ANSWER action
    answer_node_id = None
    for turn_idx, turn in enumerate(trajectory.turns):
        if turn.action and turn.action.action_type.value == 'ANSWER':
            # The node that generated the answer is from the previous turn's observation
            if turn_idx > 0:
                prev_turn = trajectory.turns[turn_idx - 1]
                if prev_turn.observation:
                    answer_node_id = id(prev_turn.observation.node)
            break

    # Collect all generated nodes by level, but stop at answer node's children
    levels = {}
    def collect_nodes(node, level=0):
        if level not in levels:
            levels[level] = []
        levels[level].append(node)

        # If this is the answer node, don't collect its children
        if answer_node_id is not None and id(node) == answer_node_id:
            return

        # Collect children (regardless of whether parent has relevance score)
        for child in node.children:
            collect_nodes(child, level + 1)

    collect_nodes(root_node)

    # Calculate positions
    max_level = max(levels.keys())
    node_positions = {}

    for level, nodes in levels.items():
        y = 1 - (level / (max_level + 1))  # Top to bottom
        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            x = (i + 1) / (num_nodes + 1)
            node_positions[id(node)] = (x, y)

    # Draw edges (plain parent-child connections)
    for level, nodes in levels.items():
        for node in nodes:
            if node.parent:
                parent_pos = node_positions[id(node.parent)]
                node_pos = node_positions[id(node)]
                ax.annotate('', xy=node_pos, xytext=parent_pos,
                          arrowprops=dict(arrowstyle='->', lw=1, color='black', alpha=0.3))

    # Draw T1 label on the root node (initial observation, no action/edge)
    if trajectory.turns and trajectory.turns[0].observation:
        root_obs_node = trajectory.turns[0].observation.node
        if id(root_obs_node) in node_positions:
            rx, ry = node_positions[id(root_obs_node)]
            ax.text(rx + 0.03, ry + 0.03, 'T1', ha='center', va='center',
                   fontsize=9, weight='bold', color='darkblue',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='blue', linewidth=1.5))

    # Draw ZOOM_IN arrows (from parent to child)
    for turn_idx, turn in enumerate(trajectory.turns):
        if turn.action and turn.action.action_type.value == 'ZOOM_IN':
            if turn.observation:
                target_node = turn.observation.node
                parent_node = target_node.parent
                if parent_node and id(parent_node) in node_positions and id(target_node) in node_positions:
                    parent_pos = node_positions[id(parent_node)]
                    target_pos = node_positions[id(target_node)]
                    ax.annotate('', xy=target_pos, xytext=parent_pos,
                              arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.7))
                    mid_x, mid_y = (parent_pos[0] + target_pos[0]) / 2, (parent_pos[1] + target_pos[1]) / 2
                    ax.text(mid_x, mid_y, f'T{turn_idx + 1}', ha='center', va='center',
                           fontsize=9, weight='bold', color='darkgreen',
                           bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='green', linewidth=1.5))

    # Draw ZOOM_OUT arrows (from child back to parent)
    for turn_idx, turn in enumerate(trajectory.turns):
        if turn.action and turn.action.action_type.value == 'ZOOM_OUT':
            # Find the previous turn to get the source node
            if turn_idx > 0:
                prev_turn = trajectory.turns[turn_idx - 1]
                if prev_turn.observation and prev_turn.observation.node.parent:
                    child_node = prev_turn.observation.node
                    parent_node = child_node.parent
                    if id(child_node) in node_positions and id(parent_node) in node_positions:
                        child_pos = node_positions[id(child_node)]
                        parent_pos = node_positions[id(parent_node)]
                        # Draw dashed arrow for zoom out
                        ax.annotate('', xy=parent_pos, xytext=child_pos,
                                  arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                                linestyle='dashed', alpha=0.7))
                        # Add turn number
                        mid_x, mid_y = (child_pos[0] + parent_pos[0]) / 2, (child_pos[1] + parent_pos[1]) / 2
                        ax.text(mid_x + 0.02, mid_y, f'T{turn_idx + 1}', ha='center', va='center',
                               fontsize=9, weight='bold', color='darkred',
                               bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='red', linewidth=1.5))

    # Draw SHIFT arrows (lateral movement between siblings)
    for turn_idx, turn in enumerate(trajectory.turns):
        if turn.action and turn.action.action_type.value == 'SHIFT':
            if turn.observation:
                target_node = turn.observation.node
                # Search backwards for the last observed sibling (same parent) as source
                source_node = None
                for prev_idx in range(turn_idx - 1, -1, -1):
                    prev_t = trajectory.turns[prev_idx]
                    if prev_t.observation:
                        candidate = prev_t.observation.node
                        if candidate.parent is target_node.parent:
                            source_node = candidate
                            break
                if source_node and id(source_node) in node_positions and id(target_node) in node_positions:
                    source_pos = node_positions[id(source_node)]
                    target_pos = node_positions[id(target_node)]
                    # Draw curved arrow for shift
                    ax.annotate('', xy=target_pos, xytext=source_pos,
                              arrowprops=dict(arrowstyle='->', lw=2, color='orange',
                                            connectionstyle='arc3,rad=-0.3', alpha=0.7))
                    # Add turn number
                    mid_x = (source_pos[0] + target_pos[0]) / 2
                    mid_y = (source_pos[1] + target_pos[1]) / 2
                    ax.text(mid_x, mid_y + 0.03, f'T{turn_idx + 1}', ha='center', va='center',
                           fontsize=9, weight='bold', color='darkorange',
                           bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='orange', linewidth=1.5))

    # Draw nodes (only nodes with relevance scores)
    for level, nodes in levels.items():
        for node in nodes:
            x, y = node_positions[id(node)]

            # Determine node color based on properties
            if node.contains_gt_most:
                color = 'lightgreen'
                edgecolor = 'darkgreen'
                linewidth = 3
            elif node.visited:
                color = 'lightblue'
                edgecolor = 'blue'
                linewidth = 2
            else:
                # Unvisited but generated node
                color = 'lightgray'
                edgecolor = 'gray'
                linewidth = 1

            # Draw node circle
            circle = plt.Circle((x, y), 0.02, color=color, ec=edgecolor, linewidth=linewidth, zorder=3)
            ax.add_patch(circle)

            # Add text: timestamp range
            time_text = f"{node.start_sec:.1f}-{node.end_sec:.1f}s"
            ax.text(x, y - 0.04, time_text, ha='center', va='top', fontsize=7)

            # Add relevance score if available
            if node.relevance_score is not None:
                score_text = f"R:{node.relevance_score:.2f}"
                ax.text(x, y + 0.04, score_text, ha='center', va='bottom',
                       fontsize=6, color='red', weight='bold')

    # Add GT interval information box
    gt_text = f'Ground Truth: [{gt_node.start_sec:.1f}s - {gt_node.end_sec:.1f}s]'
    ax.text(0.5, -0.05, gt_text, ha='center', va='center', fontsize=11, weight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='gold', edgecolor='orange', linewidth=3, alpha=0.9))

    # Configure plot
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.12, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add title
    title = f'Trajectory Tree Structure - All Generated Nodes ({len(trajectory.turns)} turns)\nQuestion: {trajectory.question[:80]}'
    if len(trajectory.question) > 80:
        title += '...'
    plt.title(title, fontsize=12, weight='bold')

    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightgreen', label='Contains GT'),
        mpatches.Patch(color='lightblue', label='Visited'),
        mpatches.Patch(color='lightgray', label='Generated (not visited)'),
        Line2D([0], [0], color='green', lw=2, marker='>', markersize=8, label='ZOOM_IN'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', marker='>', markersize=8, label='ZOOM_OUT'),
        Line2D([0], [0], color='orange', lw=2, marker='>', markersize=8, label='SHIFT')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Tree visualization saved to: {output_path}")
