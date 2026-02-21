"""
Raise Lineage

Track feature dependencies and derivation relationships.
Supports cross-organization lineage tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raise_.models.feature import Feature


@dataclass
class FeatureReference:
    """
    A reference to a feature, possibly in another organization.

    Supports both relative and absolute (cross-org) references.

    Path resolution rules:
        - "feature" - same feature group
        - "group.feature" - same project
        - "project/group.feature" - same domain
        - "domain/project/group.feature" - same org
        - "@org/domain/project/group.feature" - cross-org (absolute)
    """

    name: str
    qualified_name: str
    org: str
    domain: str
    project: str
    feature_group: str
    version: str | None = None

    @property
    def is_cross_org(self) -> bool:
        """Check if this is a cross-organization reference."""
        return self.qualified_name.startswith("@")

    @property
    def path(self) -> str:
        """Return the full path including version if present."""
        if self.version:
            return f"{self.qualified_name}@{self.version}"
        return self.qualified_name

    @classmethod
    def parse(cls, reference: str, context: dict[str, str]) -> FeatureReference:
        """
        Parse a reference string into a FeatureReference.

        Args:
            reference: The reference string to parse.
            context: Current context with org, domain, project, feature_group.

        Returns:
            A FeatureReference instance.
        """
        # Handle version suffix
        version = None
        if "@" in reference and not reference.startswith("@"):
            reference, version = reference.rsplit("@", 1)

        # Cross-org absolute reference
        if reference.startswith("@"):
            parts = reference[1:].split("/")
            if len(parts) < 4:
                raise ValueError(f"Invalid cross-org reference: {reference}")
            org = parts[0]
            domain = parts[1]
            project = parts[2]
            group_feature = parts[3]
            if "." in group_feature:
                group, name = group_feature.rsplit(".", 1)
            else:
                group = group_feature
                name = parts[4] if len(parts) > 4 else group_feature
            return cls(
                name=name,
                qualified_name=reference,
                org=org,
                domain=domain,
                project=project,
                feature_group=group,
                version=version,
            )

        # Relative reference - resolve based on context
        parts = reference.replace("/", ".").split(".")

        if len(parts) == 1:
            # Same group
            return cls(
                name=parts[0],
                qualified_name=f"{context['org']}/{context['domain']}/{context['project']}/{context['feature_group']}.{parts[0]}",
                org=context["org"],
                domain=context["domain"],
                project=context["project"],
                feature_group=context["feature_group"],
                version=version,
            )
        elif len(parts) == 2:
            # Same project: group.feature
            return cls(
                name=parts[1],
                qualified_name=f"{context['org']}/{context['domain']}/{context['project']}/{parts[0]}.{parts[1]}",
                org=context["org"],
                domain=context["domain"],
                project=context["project"],
                feature_group=parts[0],
                version=version,
            )
        elif len(parts) == 3:
            # Same domain: project/group.feature
            return cls(
                name=parts[2],
                qualified_name=f"{context['org']}/{context['domain']}/{parts[0]}/{parts[1]}.{parts[2]}",
                org=context["org"],
                domain=context["domain"],
                project=parts[0],
                feature_group=parts[1],
                version=version,
            )
        elif len(parts) == 4:
            # Same org: domain/project/group.feature
            return cls(
                name=parts[3],
                qualified_name=f"{context['org']}/{parts[0]}/{parts[1]}/{parts[2]}.{parts[3]}",
                org=context["org"],
                domain=parts[0],
                project=parts[1],
                feature_group=parts[2],
                version=version,
            )
        else:
            raise ValueError(f"Invalid reference format: {reference}")

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "org": self.org,
            "domain": self.domain,
            "project": self.project,
            "feature_group": self.feature_group,
            "version": self.version,
        }


@dataclass
class Lineage:
    """
    Lineage information for a feature.

    Tracks both upstream (dependencies) and downstream (dependents) features.
    """

    feature: Feature
    upstream: list[Feature] = field(default_factory=list)
    downstream: list[Feature] = field(default_factory=list)

    def all_upstream(self, include_external: bool = False) -> list[Feature]:
        """
        Get all transitive upstream dependencies.

        Args:
            include_external: Include features from other organizations.

        Returns:
            List of all upstream features (transitive closure).
        """
        visited = set()
        result = []

        def traverse(features: list[Feature]):
            for f in features:
                if f.qualified_name in visited:
                    continue
                if not include_external and f.org != self.feature.org:
                    continue
                visited.add(f.qualified_name)
                result.append(f)
                if f._lineage:
                    traverse(f._lineage.upstream)

        traverse(self.upstream)
        return result

    def all_downstream(self, include_external: bool = False) -> list[Feature]:
        """
        Get all transitive downstream dependents.

        Args:
            include_external: Include features from other organizations.

        Returns:
            List of all downstream features (transitive closure).
        """
        visited = set()
        result = []

        def traverse(features: list[Feature]):
            for f in features:
                if f.qualified_name in visited:
                    continue
                if not include_external and f.org != self.feature.org:
                    continue
                visited.add(f.qualified_name)
                result.append(f)
                if f._lineage:
                    traverse(f._lineage.downstream)

        traverse(self.downstream)
        return result

    def as_graph(self) -> LineageGraph:
        """Convert to a graph representation for visualization."""
        return LineageGraph.from_lineage(self)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "feature": self.feature.qualified_name,
            "upstream": [f.qualified_name for f in self.upstream],
            "downstream": [f.qualified_name for f in self.downstream],
        }


@dataclass
class LineageGraph:
    """
    Graph representation of feature lineage for visualization.
    """

    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)

    @classmethod
    def from_lineage(cls, lineage: Lineage) -> LineageGraph:
        """Create a graph from lineage information."""
        nodes = []
        edges = []
        seen_nodes = set()

        def add_node(feature: Feature):
            if feature.qualified_name not in seen_nodes:
                seen_nodes.add(feature.qualified_name)
                nodes.append({
                    "id": feature.qualified_name,
                    "name": feature.name,
                    "org": feature.org,
                    "type": "derived" if feature.derived_from else "base",
                })

        # Add center node
        add_node(lineage.feature)

        # Add upstream nodes and edges
        for upstream in lineage.all_upstream(include_external=True):
            add_node(upstream)
            edges.append({
                "source": upstream.qualified_name,
                "target": lineage.feature.qualified_name,
            })

        # Add downstream nodes and edges
        for downstream in lineage.all_downstream(include_external=True):
            add_node(downstream)
            edges.append({
                "source": lineage.feature.qualified_name,
                "target": downstream.qualified_name,
            })

        return cls(nodes=nodes, edges=edges)

    def to_ascii(self) -> str:
        """Generate an ASCII representation of the graph."""
        lines = []

        # Find upstream and downstream
        center = None
        upstream = []
        downstream = []

        for node in self.nodes:
            is_source = any(e["source"] == node["id"] for e in self.edges)
            is_target = any(e["target"] == node["id"] for e in self.edges)

            if is_source and is_target:
                center = node
            elif is_source:
                upstream.append(node)
            elif is_target:
                downstream.append(node)
            else:
                center = node

        if not center:
            return "Empty graph"

        # Render
        max_width = max(len(n["name"]) for n in self.nodes) + 4

        for u in upstream:
            lines.append(f"{u['name']:>{max_width}} ─┐")

        if upstream:
            lines.append(f"{'':>{max_width}} ├──▶ {center['name']}")
        else:
            lines.append(f"{'':>{max_width}}      {center['name']}")

        for i, d in enumerate(downstream):
            if i == 0 and upstream:
                lines.append(f"{'':>{max_width}} └──▶ {d['name']}")
            else:
                lines.append(f"{'':>{max_width}} ───▶ {d['name']}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)
