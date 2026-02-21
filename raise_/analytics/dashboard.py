"""
Raise Dashboards

Visual dashboards with parameterized charts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from raise_.analytics.analysis import Analysis
    from raise_.analytics.live_table import LiveTable


class ChartType(Enum):
    """Supported chart types."""

    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PIE = "pie"
    AREA = "area"
    TABLE = "table"
    METRIC = "metric"  # Single big number
    GAUGE = "gauge"
    BOX = "box"
    VIOLIN = "violin"


class ParameterType(Enum):
    """Dashboard parameter types."""

    DATE_RANGE = "date_range"
    SINGLE_DATE = "single_date"
    DROPDOWN = "dropdown"
    MULTI_SELECT = "multi_select"
    TEXT = "text"
    NUMBER = "number"
    SLIDER = "slider"


@dataclass
class DashboardParameter:
    """
    A parameter that can be used to filter dashboard data.

    Parameters allow users to interactively filter and explore data.

    Attributes:
        name: Parameter identifier used in queries.
        label: Display label for the parameter.
        type: Parameter type.
        default: Default value.
        options: Available options for dropdown/multi_select.
        min_value: Minimum value for number/slider.
        max_value: Maximum value for number/slider.
        feature: Feature to derive options from (for dynamic dropdowns).

    Examples:
        >>> DashboardParameter(
        ...     name="tier",
        ...     label="User Tier",
        ...     type=ParameterType.DROPDOWN,
        ...     options=["bronze", "silver", "gold", "platinum"],
        ...     default="gold",
        ... )
    """

    name: str
    label: str
    type: ParameterType
    default: Any = None
    options: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    feature: str | None = None  # For dynamic options from feature values

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type.value,
            "default": self.default,
            "options": self.options,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "feature": self.feature,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DashboardParameter:
        return cls(
            name=data["name"],
            label=data["label"],
            type=ParameterType(data["type"]),
            default=data.get("default"),
            options=data.get("options"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            step=data.get("step"),
            feature=data.get("feature"),
        )

    @classmethod
    def date_range(
        cls,
        name: str = "date_range",
        label: str = "Date Range",
        default: tuple[str, str] | None = None,
    ) -> DashboardParameter:
        """Create a date range parameter."""
        return cls(
            name=name,
            label=label,
            type=ParameterType.DATE_RANGE,
            default=list(default) if default else None,
        )

    @classmethod
    def dropdown(
        cls,
        name: str,
        label: str,
        options: list[Any],
        default: Any = None,
    ) -> DashboardParameter:
        """Create a dropdown parameter."""
        return cls(
            name=name,
            label=label,
            type=ParameterType.DROPDOWN,
            options=options,
            default=default or options[0] if options else None,
        )

    @classmethod
    def multi_select(
        cls,
        name: str,
        label: str,
        options: list[Any],
        default: list[Any] | None = None,
    ) -> DashboardParameter:
        """Create a multi-select parameter."""
        return cls(
            name=name,
            label=label,
            type=ParameterType.MULTI_SELECT,
            options=options,
            default=default or [],
        )

    @classmethod
    def slider(
        cls,
        name: str,
        label: str,
        min_value: float,
        max_value: float,
        default: float | None = None,
        step: float = 1,
    ) -> DashboardParameter:
        """Create a slider parameter."""
        return cls(
            name=name,
            label=label,
            type=ParameterType.SLIDER,
            min_value=min_value,
            max_value=max_value,
            default=default or min_value,
            step=step,
        )


@dataclass
class ChartSpec:
    """
    Generic chart specification.

    A renderer-agnostic specification for charts that can be
    translated to Plotly, Altair, Matplotlib, etc.

    Attributes:
        chart_type: Type of chart.
        x: X-axis field.
        y: Y-axis field(s).
        color: Color/grouping field.
        size: Size field (for scatter).
        facet: Faceting field.
        aggregation: Aggregation for the chart.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        legend: Show legend.
        width: Chart width.
        height: Chart height.
    """

    chart_type: ChartType
    x: str | None = None
    y: str | list[str] | None = None
    color: str | None = None
    size: str | None = None
    facet: str | None = None
    aggregation: str | None = None  # sum, avg, count, etc.

    # Styling
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    legend: bool = True
    width: int | None = None
    height: int | None = None
    color_scheme: str | None = None

    # For metric charts
    value_format: str | None = None  # e.g., ".2f", ".0%", "$,.0f"
    comparison_value: float | None = None
    comparison_label: str | None = None

    def to_dict(self) -> dict:
        return {
            "chart_type": self.chart_type.value,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "size": self.size,
            "facet": self.facet,
            "aggregation": self.aggregation,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "legend": self.legend,
            "width": self.width,
            "height": self.height,
            "color_scheme": self.color_scheme,
            "value_format": self.value_format,
            "comparison_value": self.comparison_value,
            "comparison_label": self.comparison_label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChartSpec:
        return cls(
            chart_type=ChartType(data["chart_type"]),
            x=data.get("x"),
            y=data.get("y"),
            color=data.get("color"),
            size=data.get("size"),
            facet=data.get("facet"),
            aggregation=data.get("aggregation"),
            title=data.get("title"),
            x_label=data.get("x_label"),
            y_label=data.get("y_label"),
            legend=data.get("legend", True),
            width=data.get("width"),
            height=data.get("height"),
            color_scheme=data.get("color_scheme"),
            value_format=data.get("value_format"),
            comparison_value=data.get("comparison_value"),
            comparison_label=data.get("comparison_label"),
        )


@dataclass
class Chart:
    """
    A chart in a dashboard.

    Charts can be backed by an analysis, a live table, or raw data.

    Attributes:
        title: Chart title.
        analysis: Analysis that provides data.
        live_table: Live table name (alternative to analysis).
        chart_type: Type of visualization.
        spec: Detailed chart specification.
        position: Grid position (row, col, width, height).
        refresh_interval: Auto-refresh interval in seconds.

    Examples:
        >>> Chart(
        ...     title="Daily Clicks",
        ...     analysis=Aggregation(feature="clicks", metrics=["sum"]),
        ...     chart_type=ChartType.LINE,
        ...     x="date",
        ...     y="sum",
        ... )
    """

    title: str
    analysis: Analysis | None = None
    live_table: str | None = None
    chart_type: ChartType = ChartType.LINE
    x: str | None = None
    y: str | list[str] | None = None
    color: str | None = None
    size: str | None = None

    # Layout
    row: int = 0
    col: int = 0
    width: int = 6  # Grid units (out of 12)
    height: int = 4

    # Behavior
    refresh_interval: int | None = None  # seconds

    # Internal
    _id: str = field(default="", repr=False)
    _spec: ChartSpec | None = field(default=None, repr=False)

    def __post_init__(self):
        if not self.analysis and not self.live_table:
            raise ValueError("Chart requires either analysis or live_table")

        from raise_.analytics.result import generate_id
        if not self._id:
            self._id = generate_id("chart")

        # Build spec
        self._spec = ChartSpec(
            chart_type=self.chart_type,
            x=self.x,
            y=self.y,
            color=self.color,
            size=self.size,
            title=self.title,
        )

    @property
    def spec(self) -> ChartSpec:
        """Get the chart specification."""
        return self._spec

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "title": self.title,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "live_table": self.live_table,
            "spec": self._spec.to_dict() if self._spec else None,
            "position": {
                "row": self.row,
                "col": self.col,
                "width": self.width,
                "height": self.height,
            },
            "refresh_interval": self.refresh_interval,
        }


@dataclass
class Dashboard:
    """
    A collection of charts with parameters.

    Dashboards provide a visual interface for exploring feature data.

    Attributes:
        name: Dashboard name.
        description: Human-readable description.
        charts: List of charts.
        parameters: Dashboard parameters for filtering.
        created_at: When the dashboard was created.
        updated_at: When the dashboard was last modified.
        layout: Layout configuration.

    Examples:
        >>> dashboard = fs.create_dashboard(
        ...     name="engagement-metrics",
        ...     description="User engagement KPIs",
        ... )
        >>> dashboard.add_chart(Chart(...))
        >>> dashboard.add_parameter(DashboardParameter.date_range())
        >>> url = dashboard.publish()
    """

    name: str
    description: str | None = None
    qualified_name: str = ""
    charts: list[Chart] = field(default_factory=list)
    parameters: list[DashboardParameter] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    layout: dict[str, Any] = field(default_factory=dict)

    # Internal
    _client: Any = field(default=None, repr=False)
    _published_url: str | None = field(default=None, repr=False)

    def add_chart(self, chart: Chart) -> Dashboard:
        """
        Add a chart to the dashboard.

        Args:
            chart: The chart to add.

        Returns:
            Self for chaining.
        """
        self.charts.append(chart)
        self.updated_at = datetime.now()
        return self

    def remove_chart(self, chart_id: str) -> bool:
        """
        Remove a chart by ID.

        Args:
            chart_id: The chart ID to remove.

        Returns:
            True if chart was removed.
        """
        original_len = len(self.charts)
        self.charts = [c for c in self.charts if c._id != chart_id]
        if len(self.charts) < original_len:
            self.updated_at = datetime.now()
            return True
        return False

    def add_parameter(self, parameter: DashboardParameter) -> Dashboard:
        """
        Add a parameter to the dashboard.

        Args:
            parameter: The parameter to add.

        Returns:
            Self for chaining.
        """
        self.parameters.append(parameter)
        self.updated_at = datetime.now()
        return self

    def remove_parameter(self, name: str) -> bool:
        """
        Remove a parameter by name.

        Args:
            name: The parameter name to remove.

        Returns:
            True if parameter was removed.
        """
        original_len = len(self.parameters)
        self.parameters = [p for p in self.parameters if p.name != name]
        if len(self.parameters) < original_len:
            self.updated_at = datetime.now()
            return True
        return False

    def render(
        self,
        format: str = "html",
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """
        Render the dashboard.

        Args:
            format: Output format (html, json, png).
            parameters: Parameter values for filtering.

        Returns:
            Rendered dashboard content.
        """
        if format == "json":
            return self._render_json(parameters)
        elif format == "html":
            return self._render_html(parameters)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _render_json(self, parameters: dict[str, Any] | None = None) -> str:
        """Render as JSON specification."""
        import json

        spec = {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "parameter_values": parameters or {},
            "charts": [c.to_dict() for c in self.charts],
            "layout": self.layout,
        }
        return json.dumps(spec, indent=2, default=str)

    def _render_html(self, parameters: dict[str, Any] | None = None) -> str:
        """Render as HTML."""
        # Generate a simple HTML template
        # In production, this would use a proper template engine
        charts_html = []
        for chart in self.charts:
            charts_html.append(f"""
                <div class="chart" style="grid-column: span {chart.width}; grid-row: span {chart.height};">
                    <h3>{chart.title}</h3>
                    <div class="chart-container" data-spec='{chart.to_dict()}'></div>
                </div>
            """)

        params_html = []
        for param in self.parameters:
            params_html.append(f"""
                <div class="parameter">
                    <label>{param.label}</label>
                    <input type="text" name="{param.name}" value="{param.default or ''}">
                </div>
            """)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.name}</title>
            <style>
                body {{ font-family: system-ui, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; }}
                .chart {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
                .parameters {{ display: flex; gap: 20px; margin-bottom: 20px; }}
                .parameter label {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>{self.name}</h1>
            <p>{self.description or ''}</p>
            <div class="parameters">
                {''.join(params_html)}
            </div>
            <div class="dashboard">
                {''.join(charts_html)}
            </div>
            <script>
                // Chart rendering would be implemented here
                // using the chart specs to render with Plotly, Vega, etc.
            </script>
        </body>
        </html>
        """

    def publish(self) -> str:
        """
        Publish the dashboard and return its URL.

        Returns:
            URL where the dashboard can be accessed.
        """
        # In production, this would upload to a hosting service
        from raise_.analytics.result import generate_id

        if not self._published_url:
            dashboard_id = generate_id("dash")
            self._published_url = f"https://dashboards.raise.dev/{dashboard_id}"

        return self._published_url

    def unpublish(self) -> None:
        """Unpublish the dashboard."""
        self._published_url = None

    def delete(self) -> None:
        """Delete this dashboard."""
        if self._client:
            self._client.delete_dashboard(self.name)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "description": self.description,
            "charts": [c.to_dict() for c in self.charts],
            "parameters": [p.to_dict() for p in self.parameters],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "layout": self.layout,
            "published_url": self._published_url,
        }

    def __repr__(self) -> str:
        return (
            f"Dashboard(name={self.name!r}, charts={len(self.charts)}, "
            f"parameters={len(self.parameters)})"
        )
