import numpy as np
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_timeline(data: np.array = []) -> None:
    assert isinstance(data, np.ndarray) or (
        len(data) > 0 and len(data[0]) > 0
    ), "Wrong data type"

    df = pd.DataFrame(
        columns=["Truck", "StartTime", "FinishTime", "StartNode", "EndNode", "Trailer"]
    )

    trailer_names = []
    for start_node, target_node, truck, trailer, time in data:
        if trailer == -1:
            trailer = "Trailer {}".format(None)
        else:
            trailer = "Trailer {}".format(trailer)

        if truck == -1:
            truck = None
        else:
            truck = int(truck)

        if trailer not in trailer_names:
            trailer_names.append(trailer)

        block = dict(
            Truck="Truck {}".format(truck),
            StartTime=time,
            FinishTime=time + 1,
            StartNode="Node {}".format(start_node),
            EndNode="Node {}".format(target_node),
            Trailer=trailer,
        )
        df = df.append(block, ignore_index=True)

    df["StartTime"] = pd.to_datetime(df["StartTime"], unit="h")
    df["FinishTime"] = pd.to_datetime(df["FinishTime"], unit="h")

    fig = go.Figure()

    # Generate a categorical color scale based on the unique trailer names
    trailer_colors = px.colors.qualitative.Set3[: len(trailer_names)]

    for truck, truck_df in df.groupby("Truck"):
        for idx, row in truck_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["StartTime"], row["FinishTime"]],
                    y=[truck, truck],
                    mode="lines",
                    name=truck,
                    line=dict(
                        width=20,
                        color=trailer_colors[trailer_names.index(row["Trailer"])],
                    ),
                    hoverinfo="none",
                    showlegend=False,
                )
            )
            fig.add_annotation(
                x=row["StartTime"],
                y=truck,
                xref="x",
                yref="y",
                showarrow=False,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=1.5,
                ax=20,
                ay=0,
                text=row["StartNode"],
                font=dict(size=8),
                align="center",
                valign="bottom",
            )

            fig.add_annotation(
                x=row["FinishTime"],
                y=truck,
                xref="x",
                yref="y",
                showarrow=False,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=1.5,
                ax=-20,
                ay=0,
                text=row["EndNode"],
                font=dict(size=8),
                align="center",
                valign="bottom",
            )

    # Add legend for the trailer colors
    legend_data = []
    for i, trailer_name in enumerate(trailer_names):
        legend_data.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=trailer_colors[i]),
                name=trailer_name,
                legendgroup=trailer_name,
            )
        )

    fig.add_traces(legend_data)
    fig.update_layout(legend=dict(traceorder="reversed"), showlegend=True)
    fig.update_xaxes(showticklabels=False)

    fig.update_yaxes(autorange="reversed")  # Tasks listed from top to bottom
    fig.show()


if __name__ == "__main__":
    data = torch.tensor(
        [
            [3.0, 2.0, 0.0, -1.0, 0.0],
            [3.0, 1.0, 1.0, -1.0, 1.0],
            [2.0, 1.0, 0.0, 0.0, 2.0],
            [1.0, 2.0, 1.0, 1.0, 3.0],
            [1.0, 2.0, 0.0, 2.0, 4.0],
            [2.0, 2.0, 1.0, -1.0, 5.0],
            [2.0, 2.0, 0.0, -1.0, 6.0],
            [2.0, 2.0, 1.0, -1.0, 7.0],
            [2.0, 2.0, 0.0, -1.0, 8.0],
        ]
    )
    data = data.numpy().astype(int)
    plot_timeline(data)
