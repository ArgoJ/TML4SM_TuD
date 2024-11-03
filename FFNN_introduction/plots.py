import numpy as np
import tensorflow as tf

from keras import Model
from scipy.interpolate import griddata
from bokeh.plotting import figure, column
from bokeh.palettes import Category20_20, Sunset10
from bokeh.models import Legend, ColumnDataSource, HoverTool, RadioButtonGroup, CustomJS, LinearColorMapper, ColorBar, ContourColorBar
from bokeh.models.layouts import Column

from plot_utils import show__hide_all_button, set_figure_to_default_latex



def change_range_select(fig: figure, max_epochs: set[int]) -> RadioButtonGroup:
    max_epochs = sorted(list(max_epochs))
    range_labels = [f'{e} epochs' for e in max_epochs]
    select = RadioButtonGroup(labels=range_labels, active=len(max_epochs)-1)
    callback = CustomJS(args=dict(fig=fig, max_epochs=max_epochs), code="""
        var index = cb_obj.active;
        var epoch_value = max_epochs[index];
        
        // Update the x_range of the figure based on the selected epoch
        fig.x_range.start = 0;
        fig.x_range.end = epoch_value;
    """)
    select.js_on_change('active', callback)
    return select



def plot_loss_history(
        loss_history: dict[str, list[float]], 
        width: int = 1200, 
        height: int = 1000, 
        colors = Category20_20, 
        use_latex_style: bool = False
    ) -> None:

    p = figure(
        title="Model Loss History",
        x_axis_label="Epoch",
        y_axis_label="log10 MSE",
        y_axis_type="log",
        width=width,
        height=height,
    )

    legend_items = []
    renderers = []
    all_epochs = []
    for i, (model_name, losses) in enumerate(loss_history.items()):
        total_epochs = list(range(len(losses)))
        min_los = min(losses)
        min_epoch = losses.index(min_los)

        all_epochs.append(len(losses))

        data = {
            'epoch': total_epochs, 
            'loss': losses,
            'minimal_loss': [min_los for _ in range(len(losses))],
            'minimal_epoch': [min_epoch for _ in range(len(losses))],
        }
        datasource = ColumnDataSource(data=data)

        line = p.line(
            x='epoch',
            y='loss',
            source=datasource,
            line_width=2,
            color=colors[i % len(colors)],
        )
        renderers.append(line)
        legend_items.append((model_name, [line]))

    hover = HoverTool(tooltips=[("minimal loss", "@minimal_loss")], mode='mouse', renderers=renderers)
    p.add_tools(hover)

    legend = Legend(items=legend_items, click_policy='hide', orientation='horizontal')
    legend.nrows = 8
    p.add_layout(legend, 'below')

    hide_select = show__hide_all_button(renderers)
    range_select = change_range_select(p, set(all_epochs))
    layout = column([hide_select, range_select, p])

    if use_latex_style:
        set_figure_to_default_latex(layout, move_legend_top=False, output_backend='svg')
    return layout




def plot_x_y(
        predictions: dict[str, tf.Tensor], 
        xs, ys, xs_c, ys_c, 
        width: int = 1200, 
        height: int = 1000,
        colors = Category20_20,
        use_latex_style: bool = False,
        title: str | None = "Model Prediction"
    ) -> None:

    p = figure(
        title=title,
        x_axis_label="x",
        y_axis_label="y",
        width=width,
        height=height,
    )
    legend_items = []
    renderers = []

    # Calibration data
    calibration_data = {
        'x': xs_c, 
        'y': ys_c,
    }
    cal_datasource = ColumnDataSource(data=calibration_data)
    cal_scatter = p.scatter(
        x='x',
        y='y',
        source=cal_datasource,
        size=6,
        color=colors[0],
    )
    renderers.append(cal_scatter)
    legend_items.append(('Calibration Data', [cal_scatter]))

    # Bathtub function
    true_data = {
        'x': xs, 
        'y': ys,
    }
    true_datasource = ColumnDataSource(data=true_data)
    true_line = p.line(
        x='x',
        y='y',
        source=true_datasource,
        line_width=4,
        color=colors[1],
    )
    renderers.append(true_line)
    legend_items.append(('Ground Truth', [true_line]))

    # Other models
    if predictions is not None and predictions:
        
        for i, (model_name, preds) in enumerate(predictions.items()):
            datasource = ColumnDataSource(data = {'x': xs, 'y': preds})
            line = p.line(
                x='x',
                y='y',
                source=datasource,
                line_width=4,
                color=colors[(i+2) % len(colors)],
            )
            renderers.append(line)
            legend_items.append((model_name, [line]))

    # Add Hover tool
    hover = HoverTool(tooltips=[("x, y", "@x, @y")], mode='mouse', renderers=renderers)
    p.add_tools(hover)

    # Add Legend
    legend = Legend(items=legend_items, click_policy='hide', orientation='horizontal')
    legend.nrows = 8
    p.add_layout(legend, 'below')

    # Add hide select button
    hide_select = show__hide_all_button(renderers)
    layout = column([hide_select, p])

    if use_latex_style:
        set_figure_to_default_latex(layout, move_legend_top=False, output_backend='svg')
    return layout



def plot_x_y_z(
        predictions: dict[str, tf.Tensor], 
        xs: tf.Tensor, 
        ys: tf.Tensor, 
        zs: tf.Tensor, 
        width: int = 1000, 
        height: int = 800,
        contour_colors = Sunset10,
        use_latex_style: bool = False,
        title: str = "Model Predictions with Ground Truth Contours"
    ) -> Column:
    # Set up Bokeh figure
    p = figure(
        title=title,
        x_axis_label="x",
        y_axis_label="y",
        width=width,
        height=height,
        tools="pan,wheel_zoom,reset",
    )
    legend_items = []
    renderers = []

    # Plot ground truth as contours
    n = int(np.sqrt(len(xs)))
    contour_source ={
        'x': np.reshape(xs, (n, n)), 
        'y': np.reshape(ys, (n, n)), 
        'z': np.reshape(zs, (n, n)), 
    }
    levels = np.linspace(np.min(zs), np.max(zs), 10)
    contour_renderer = p.contour(
        x=contour_source['x'], 
        y=contour_source['y'], 
        z=contour_source['z'], 
        levels=levels, 
        fill_color=contour_colors,
        line_color="black",
    )
    contour_color_bar = contour_renderer.construct_color_bar(title="Ground Truth")

    # Add color bar for contour
    p.add_layout(contour_color_bar, 'right')

    

    if predictions is not None and predictions:
        # Plot model predictions as scatter points with color mapping
        all_preds = np.concatenate([pred for pred in predictions.values()])
        min_pred = np.min(all_preds)
        max_pred = np.max(all_preds)

        pred_color_mapper = LinearColorMapper(palette="Plasma256", low=min_pred, high=max_pred)
        for model_name, preds in predictions.items():
            pred_source = ColumnDataSource(data = {'x': xs, 'y': ys, 'z': preds})
            pred_renderer = p.scatter(
                x='x',
                y='y',
                size=6,
                source=pred_source,
                color={'field': 'z', 'transform': pred_color_mapper},
                alpha=0.6
            )
            legend_items.append((model_name, [pred_renderer]))
            renderers.append(pred_renderer)

        # Add color bar for model predictions
        color_bar = ColorBar(color_mapper=pred_color_mapper, location=(0, 0), title="Model Prediction")
        p.add_layout(color_bar, 'right')

    # Add hover tool to show x, y, z values
    hover = HoverTool(tooltips=[("x, y, z", "@x, @y, @z")])
    p.add_tools(hover)

    # Add Legend
    legend = Legend(items=legend_items, click_policy='hide', orientation='horizontal')
    legend.nrows = 8
    p.add_layout(legend, 'below')

    # Add hide select button
    hide_select = show__hide_all_button(renderers)
    layout = column([hide_select, p])

    if use_latex_style:
        set_figure_to_default_latex(layout, move_legend_top=False, output_backend='svg')
    return layout
