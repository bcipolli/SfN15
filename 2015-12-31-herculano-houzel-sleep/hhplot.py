from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import scipy.stats
from bokeh.plotting import figure, show, output_notebook

# Create a colormap, as used in Figure 2
colormap = dict(Primate='red', Eulipotyphla='yellow', Glires='green',
                Afrotheria='blue', Aritodactyla='cyan', Scandentia='black')
labels = {
    'D/A': 'density/area (N mg^{-1}mm^{-2})',
    'A/D': 'area/density (mg mm^2/N)',
    'T': 'thickness (mm)',
    'Acx': 'surface area (mm^2)',
    'Ncx': 'neurons',
    'Mcx': 'cortical mass (g)',
    'brain mass': 'brain mass (g)',
    'Dcx': 'density (N mg^{-1})',
    'O/N': 'other cells/neurons',
    'N/A': 'neurons/area (mm^{-2})',
    'Vcx': 'grey matter volume (mm^3)'}

plot_lib = 'mpl'


def set_plot_lib(lib):
    global plot_lib
    plot_lib = lib


def show_plot(p, lib=None, output_dir=None):
    lib = lib or plot_lib

    def get_fig_name(fig, fi=None):
        default_val = 'figure' + (str(fi) if fi is not None else '')
        return getattr(fig, 'name', default_val)

    if output_dir is None:
        if lib == 'bokeh':
            show(p)
    elif lib == 'mpl':
        # Save to png
        for fi in plt.get_fignums():
            filename = '%s.png' % get_fig_name(plt.figure(fi), fi)
            plt.figure(fi)
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
    elif lib == 'bokeh':
        filename = '%s.html' % get_fig_name(p)
        output_file(os.path.join(output_dir), title=get_fig_name(p))


def get_data(key, data):
    # Note that in figure 1, sleep and thickness
    # are NOT log10. Code that here, so it's clean later.
    if key in ['T', 'daily sleep']:
        return data[key]
    else:
        return np.log10(data[key])


def scatter(xkey, ykey, data, lib=None):
    # Helper function to do a scatter plot and then label things.
    lib = lib or plot_lib

    xvals = get_data(xkey, data=data)
    yvals = get_data(ykey, data=data)
    colors = [colormap[cls] for cls in data['Order']]
    if lib == 'mpl':
        ax = plt.gca()
        ax.scatter(xvals, yvals, c=colors, s=100)  # size is hardcoded
        ax.set_xlabel(labels.get(xkey, xkey))
        ax.set_ylabel(labels.get(ykey, ykey))
    else:
        from bokeh.plotting import ColumnDataSource
        from bokeh.models import HoverTool

        source = ColumnDataSource(
            data = dict(
                x=xvals,
                y=yvals,
                color=colors,
                order=data['Order'],
                species=data['Species']))

        p = figure(tools="crosshair,pan,reset,save")
        import bokeh.models as bkm
        import bokeh.plotting as bkp
        p.xaxis.axis_label = labels.get(xkey, xkey)
        p.yaxis.axis_label = labels.get(ykey, ykey)
        g1 = bkm.Circle(x='x', y='y', fill_color='color', size=25)
        g1_r = p.add_glyph(source_or_glyph=source, glyph=g1)
        g1_hover = bkm.HoverTool(renderers=[g1_r],
                                 point_policy='follow_mouse',
                                 tooltips=OrderedDict([
            ("Species", "@species"),
            ("Order", "@order"),
            ("(%s, %s)" % (xkey, ykey), "($x, $y)"),
        ]))
        p.add_tools(g1_hover)
        ax = p
    return ax


def regress_and_plot(key1, key2, data, lib=None):
    # Helper function to do the regression,
    # show the regression line, and show the rvalue.
    lib = lib or plot_lib

    # Select rows without NaNs
    xvals = get_data(key1, data=data)
    yvals = get_data(key2, data=data)
    bad_idx = np.isnan(xvals + yvals)
    xvals, yvals = xvals[~bad_idx], yvals[~bad_idx]

    # Do the regression
    res = scipy.stats.linregress(xvals, yvals)

    # Plot with regression line and text.
    ax = scatter(key1, key2, data=data, lib=lib)
    xlims = np.array([np.min(xvals), np.max(xvals)])
    ylims = np.array([np.min(yvals), np.max(yvals)])
    lbl = ('%.3f' if res.pvalue < 0.01 else 'n.s. (%.3f)') % res.rvalue
    if lib == 'mpl':
        ax.plot(xlims, res.slope * xlims + res.intercept)
        ax.text(xlims[0 if res.rvalue > 0 else 1],
                ylims[1], lbl)
    else:
        # ax.ray(x=[0], y=[res.intercept], length=0, angle=np.pi/2-np.arctan(res.slope), line_width=1)
        ax.line(xlims, res.slope * xlims + res.intercept)
        ax.text(x=xlims[0] if res.rvalue > 0 else xlims[1] - 0.2*(xlims[1] - xlims[0]),
                y=0.85*ylims[1], text=[lbl])
    return ax


def grid_it(all_xkeys, data, fn=regress_and_plot, lib=None):
    # Make a grid of plots; a bit cleaner than just plotting one-by-one.
    lib = lib or plot_lib

    all_xkeys = np.asarray(all_xkeys).T
    n_rows, n_cols = all_xkeys.shape
    for pi in range(all_xkeys.size):
        if lib == 'mpl':
            if pi == 0:
                plt.figure(figsize=(12, 12))
            plt.subplot(n_rows, n_cols, (pi + 1) % all_xkeys.size)
        p = fn(all_xkeys.ravel()[pi], 'daily sleep', data=data, lib=lib)
        if lib == 'bokeh':
            p.plot_height = p.plot_width = 300
            if (pi+1) % n_cols == 1:
                if pi == 0:
                    plots = []
                row = []
            row.append(p)
            if (pi+1) % n_cols == 0:
                plots.append(row)
    if lib == 'bokeh':
        from bokeh.models import GridPlot
        g = GridPlot(children=plots)
        g.plot_width = g.plot_height = 1200
        return g


def do_pca(cols, data, zscore=True, n_components=2):
    # I'm interested in the PCA analysis.
    # This function allows me to select some of the data,
    # run PCA with it, and get some summary info.
    #
    # I can choose to zscore each column (to standardize scores)
    # n_components is 2 by default, as that's what's in the paper.
    import sklearn.decomposition
    pca = sklearn.decomposition.PCA(whiten=False, n_components=n_components or len(cols))

    pca_data = np.asarray([get_data(col, data=data) for col in cols]).T
    print pca_data.shape
    idx = np.isnan(pca_data).sum(axis=1) == 0
    pca_data = pca_data[idx]  # remove nan
    if zscore:
        for k, col in enumerate(pca_data.T):
            pca_data[:, k] = scipy.stats.mstats.zscore(col)
    res = pca.fit_transform(pca_data)
    print cols
    print 'Total variance explained: ', pca.explained_variance_ratio_.sum()
    print 'Variance explained per component', pca.explained_variance_ratio_
    print pca.components_[0:2]
    print ''


def lin_regress(cols, predict_col, data, zscore=True):
    # We are interested in one value. What about linear regression?
    import sklearn.linear_model
    lm = sklearn.linear_model.LinearRegression(normalize=False)
    lm_data = np.asarray([get_data(col, data=data) for col in cols if cols != predict_col]).T
    yvals = np.asarray(get_data(predict_col, data=data))

    idx = np.isnan(lm_data).sum(axis=1) == 0
    lm_data = lm_data[idx]  # remove nan
    yvals = yvals[idx]
    print lm_data.shape

    if zscore:
        for k, col in enumerate(lm_data.T):
            lm_data[:, k] = scipy.stats.mstats.zscore(col)
        yvals = scipy.stats.mstats.zscore(yvals)

    res = lm.fit(lm_data, yvals)
    return res, lm
