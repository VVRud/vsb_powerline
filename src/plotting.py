def plot_phases(df, df_meta, measurement_id, plot_range=[0, 800000], url=False):
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
    init_notebook_mode(connected=True)
    
    rg = [i for i in range(plot_range[0], plot_range[1])]
    targets = df_meta[df_meta['id_measurement'] == measurement_id]['target'].values

    phase0 = go.Scatter(
        x = rg,
        y = df[str(measurement_id * 3)].values[plot_range[0]:plot_range[1]],
        mode = 'lines',
        name = 'phase1 = {}'.format(targets[0]),
        opacity = 0.75
    )

    phase1 = go.Scatter(
        x = rg,
        y = df[str(measurement_id * 3 + 1)].values[plot_range[0]:plot_range[1]],
        mode = 'lines',
        name = 'phase2 = {}'.format(targets[1]),
        opacity = 0.75
    )

    phase2 = go.Scatter(
        x = rg,
        y = df[str(measurement_id * 3 + 2)].values[plot_range[0]:plot_range[1]],
        mode = 'lines',
        name = 'phase3 = {}'.format(targets[2]),
        opacity = 0.75
    )
    
    layout = go.Layout(
        title='PHASES. {} damaged.'.format(sum(targets)),
        xaxis=dict(
            title='TIME',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        ),
        yaxis=dict(
            title='DATA',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        )
    )
    
    data = [phase0, phase1, phase2]
    fig = go.Figure(data=data, layout=layout)
    
    if url:
        return plot(fig, filename='plots/phases_{}.html'.format(measurement_id))
    else:
        return iplot(fig)
    
def plot_single_func(df, df_meta, measurement_id, func, name='func', plot_range=[0, 800000], url=False):
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
    init_notebook_mode(connected=True)
    
    rg = [i for i in range(plot_range[0], plot_range[1])]
    targets = df_meta[df_meta['id_measurement'] == measurement_id]['target'].values
    values = func(df, measurement_id)
    
    data = go.Scatter(
        x = rg,
        y = values[plot_range[0]:plot_range[1]],
        mode = 'lines',
        name = name
    )
    
    layout = go.Layout(
        title='PHASES {}. {} damaged. ({})'.format(name.upper(), sum(targets), targets),
        xaxis=dict(
            title='TIME',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        ),
        yaxis=dict(
            title='DATA',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        )
    )
    fig = go.Figure(data=[data], layout=layout)

    if url:
        return plot(fig, filename='plots/phases_{}_{}.html'.format(name.lower(), measurement_id))
    else:
        return iplot(fig)
    
def plot_phases_func(df, df_meta, measurement_id, func, name='func', plot_range=[0, 800000], url=False):
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
    init_notebook_mode(connected=True)
    
    rg = [i for i in range(plot_range[0], plot_range[1])]
    targets = df_meta[df_meta['id_measurement'] == measurement_id]['target'].values
    values = func(df, measurement_id)

    phase0 = go.Scatter(
        x = rg,
        y = values[0][plot_range[0]:plot_range[1]],
        mode = 'lines',
        name = 'phase1 = {}'.format(targets[0]),
        opacity = 0.75
    )

    phase1 = go.Scatter(
        x = rg,
        y = values[1][plot_range[0]:plot_range[1]],
        mode = 'lines',
        name = 'phase2 = {}'.format(targets[1]),
        opacity = 0.75
    )

    phase2 = go.Scatter(
        x = rg,
        y = values[2][plot_range[0]:plot_range[1]],
        mode = 'lines',
        name = 'phase3 = {}'.format(targets[2]),
        opacity = 0.75
    )
    
    layout = go.Layout(
        title='PHASES {}. {} damaged.'.format(name.upper(), sum(targets)),
        xaxis=dict(
            title='TIME',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        ),
        yaxis=dict(
            title='DATA',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        )
    )
    
    data = [phase0, phase1, phase2]
    fig = go.Figure(data=data, layout=layout)
    
    if url:
        return plot(fig, filename='plots/phases_{}_{}.html'.format(name.lower(), measurement_id))
    else:
        return iplot(fig)
    
def plot_values(range_values, values, url=False):
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
    init_notebook_mode(connected=True)
        
    data = go.Scatter(
        x = range_values,
        y = values,
        mode = 'lines',
        name = 'SINGLE'
    )
    
    layout = go.Layout(
        title='SINGLE',
        xaxis=dict(
            title='TIME',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        ),
        yaxis=dict(
            title='DATA',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        )
    )
    fig = go.Figure(data=[data], layout=layout)

    if url:
        return plot(fig, filename='plots/plot_single.html')
    else:
        return iplot(fig)
    
def plot_values_loglog(range_values, values, url=False):
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
    init_notebook_mode(connected=True)
        
    data = go.Scatter(
        x = range_values,
        y = values,
        mode = 'lines',
        name = 'SINGLE'
    )
    
    layout = go.Layout(
        title='SINGLE',
        xaxis=dict(
            title='TIME',
            type='log',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        ),
        yaxis=dict(
            title='DATA',
            type='log',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            )
        )
    )
    fig = go.Figure(data=[data], layout=layout)

    if url:
        return plot(fig, filename='plots/plot_single.html')
    else:
        return iplot(fig)