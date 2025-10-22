
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

class UPG:
    def __init__(self):
        self.concepts = []
        self.metadata = {}
    
    def add_concept(self, name, description, x, y, z, delta, category="general"):
        """Add a new concept to the UPG"""
        concept = {
            "id": len(self.concepts),
            "name": name,
            "description": description,
            "coordinates": {"x": x, "y": y, "z": z, "delta": delta},
            "category": category,
            "relationships": []
        }
        self.concepts.append(concept)
        return concept["id"]
    
    def calculate_distance(self, concept1, concept2):
        """Calculate 4D Euclidean distance between two concepts"""
        coords1 = concept1["coordinates"]
        coords2 = concept2["coordinates"]
        return np.sqrt(
            (coords1["x"] - coords2["x"])**2 +
            (coords1["y"] - coords2["y"])**2 +
            (coords1["z"] - coords2["z"])**2 +
            (coords1["delta"] - coords2["delta"])**2
        )
    
    def find_nearest_concepts(self, concept_id, n=5):
        """Find n nearest concepts to a given concept"""
        target_concept = self.concepts[concept_id]
        distances = []
        for i, concept in enumerate(self.concepts):
            if i != concept_id:
                distance = self.calculate_distance(target_concept, concept)
                distances.append((i, distance, concept["name"]))
        distances.sort(key=lambda x: x[1])
        return distances[:n]
    
    def plot_2d(self, plane="xy", color_by="delta", size_by="z"):
        """Plot UPG concepts in 2D projection"""
        if plane == "xy":
            x_coords = [c["coordinates"]["x"] for c in self.concepts]
            y_coords = [c["coordinates"]["y"] for c in self.concepts]
            x_label, y_label = "Golden Ratio (φ)", "Silver Ratio (δ)"
        elif plane == "xz":
            x_coords = [c["coordinates"]["x"] for c in self.concepts]
            y_coords = [c["coordinates"]["z"] for c in self.concepts]
            x_label, y_label = "Golden Ratio (φ)", "Consciousness (ζ)"
        elif plane == "yz":
            x_coords = [c["coordinates"]["y"] for c in self.concepts]
            y_coords = [c["coordinates"]["z"] for c in self.concepts]
            x_label, y_label = "Silver Ratio (δ)", "Consciousness (ζ)"
        else:
            raise ValueError("Plane must be 'xy', 'xz', or 'yz'")
        
        colors = [c["coordinates"][color_by] for c in self.concepts]
        sizes = [c["coordinates"][size_by] * 100 for c in self.concepts]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=0.7)
        plt.colorbar(scatter, label=color_by.title())
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"UPG Concepts - {plane.upper()} Plane")
        
        # Add labels for each point
        for i, concept in enumerate(self.concepts):
            plt.annotate(concept["name"], (x_coords[i], y_coords[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering on UPG concepts"""
        coordinates = np.array([[c["coordinates"]["x"], c["coordinates"]["y"], 
                               c["coordinates"]["z"], c["coordinates"]["delta"]] 
                              for c in self.concepts])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # Add cluster assignments to concepts
        for i, concept in enumerate(self.concepts):
            concept["cluster"] = int(cluster_labels[i])
        
        return cluster_labels, kmeans.cluster_centers_
    
    def analyze_distributions(self):
        """Analyze coordinate distributions across all concepts"""
        x_coords = [c["coordinates"]["x"] for c in self.concepts]
        y_coords = [c["coordinates"]["y"] for c in self.concepts]
        z_coords = [c["coordinates"]["z"] for c in self.concepts]
        delta_coords = [c["coordinates"]["delta"] for c in self.concepts]
        
        return {
            "x_stats": {"mean": np.mean(x_coords), "std": np.std(x_coords)},
            "y_stats": {"mean": np.mean(y_coords), "std": np.std(y_coords)},
            "z_stats": {"mean": np.mean(z_coords), "std": np.std(z_coords)},
            "delta_stats": {"mean": np.mean(delta_coords), "std": np.std(delta_coords)}
        }
    
    def export_to_json(self, filename):
        """Export UPG data to JSON file"""
        data = {
            "concepts": self.concepts,
            "metadata": {
                "total_concepts": len(self.concepts),
                "export_date": "2024-01-01T00:00:00Z",
                "version": "1.0.0"
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_from_json(self, filename):
        """Load UPG data from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.concepts = data["concepts"]
        self.metadata = data["metadata"]
        return self
    