import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from glogic_concept_kingdon import GeometricLogic


class GeometricLogicVisualizer:
    def __init__(self, logic_system):
        self.logic_system = logic_system
        self.graph = nx.DiGraph()
        self.pos = None

    def build_graph(self):
        """Builds a directed graph representing logical relations with categorized nodes."""
        for relation, elements in self.logic_system.relations.items():
            for element in elements.__dict__.values():  # Extracting components
                self.graph.add_edge(relation, str(element), label="Relation")

        for quantifier, statement in self.logic_system.quantifiers.items():
            self.graph.add_edge(quantifier, statement, label="Quantifier")

        self.pos = nx.spring_layout(self.graph)

    def draw_graph(self):
        """Draws the logical graph with styled nodes and labels."""
        plt.figure(figsize=(10, 6))

        node_colors = []
        for node in self.graph.nodes:
            if node.startswith("∀") or node.startswith("∃"):
                node_colors.append("lightcoral")  # Quantifiers
            elif node in self.logic_system.relations:
                node_colors.append("lightgreen")  # Relations
            else:
                node_colors.append("lightblue")  # Statements

        edge_labels = {(u, v): d['label'] for u, v, d in self.graph.edges(data=True)}

        nx.draw(self.graph, self.pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=3000,
                font_size=10)
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels, font_size=8)

        plt.title("Geometric Logic System Visualization")
        plt.show()

    def animate_graph(self):
        """Animates the logical deduction process dynamically over time."""
        fig, ax = plt.subplots(figsize=(10, 6))

        def update(num):
            ax.clear()
            current_edges = list(self.graph.edges())[:num + 1]
            animated_graph = nx.DiGraph(current_edges)

            nx.draw(animated_graph, self.pos, with_labels=True, node_color="lightblue", edge_color="gray",
                    node_size=3000, font_size=10)
            nx.draw_networkx_edge_labels(animated_graph, self.pos, edge_labels=edge_labels, font_size=8)
            ax.set_title("Logical Deduction Animation Step: " + str(num + 1))

        edge_labels = {(u, v): d['label'] for u, v, d in self.graph.edges(data=True)}
        ani = animation.FuncAnimation(fig, update, frames=len(self.graph.edges()), repeat=False, interval=1000)
        plt.show()


if __name__ == "__main__":
    logic_system = GeometricLogic()
    logic_system.test_examples()

    visualizer = GeometricLogicVisualizer(logic_system)
    visualizer.build_graph()
    visualizer.draw_graph()
    visualizer.animate_graph()
