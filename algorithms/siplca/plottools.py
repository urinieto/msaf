# Copyright (C) 2009 Ron J. Weiss (ronweiss@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import functools
import types

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if mpl.rcParams['backend'].startswith('WX'):
    from plot_pages_wx import plot_pages
else:
    from plot_pages import plot_pages

def plot_on_same_axes(*args):
    plot_args = []
    for arg in args:
        plot_args.append(np.arange(len(arg)))
        plot_args.append(arg)
    plt.plot(*plot_args)

def plot_or_imshow(a, **kwargs):
    ndim = np.asarray(a).ndim
    if ndim == 1:
        h = plt.plot(a, **kwargs)
    else:
        h = plt.imshow(a, origin='lower', **kwargs)
    return h

def plotall(arrays, **props):
    """Plots all of the given arrays as subplots of the same figure.

    Supported properties applied across all subplots (default value):
    'align'                ('xyzc'): set axes to be aligned.  can be any
    combination of 'x', 'y', 'z', and 'c'
    'colorbar'              (true): if true displays a colorbar next to each plot
    'order'                  ('r'): ordering of subplots ('r' for row-major order
    or 'c' for column-major order)
    'plotfun'     (plot_or_imshow): function to use for plotting each element
    of data
    'pub'                  (false): If true, try to make nicer looking
    plots suitable for publication
    'subplot'              ([n 1]): subplot arrangement
    'clickfun'              (None):
    'transpose'            (False):
    
    If the 'pub' property is false, additional GUI controls are added to
    the figure, including scrollbars to control panning, zooming, and
    'caxis' settings.  Also, if n is larger than the number of subplots
    specified in the properties, s, then only s subplots will be
    displayed at a given time, but paging controls will be added to
    figure to give access to the remaining n-s plots.
    
    A series of name value pairs that correspond to optional settings
    and various matplotlib plot properties can optionally be passed as
    keyword arguments.
    
    Each per-subplot property value can be either a scalar, in which
    case the same value is applied to all subplots, or a function that
    takes no arguments, it will be evaluated each time the property is
    set.  For example, is useful for adjusting the units of tick
    labels without knowing where the ticks are in advance.
    E.g. setting xticklabels=lambda: gca().get_xticks()*1e-3 can
    automatically convert the horizontal axis labels from milliseconds
    to seconds.
    """
    default_props = {'align': 'xyzc',
                     'clf': True,
                     'clickfun': None,
                     'colorbar': True,
                     'grid': True,
                     'order': 'r',
                     'subplot': (len(arrays), 1),
                     'plotfun': plot_or_imshow,
                     'pub': False,
                     'transpose': False}
    for k,v in default_props.iteritems():
        if not k in props:
            props[k] = v

    other_props = {}
    for k,v in props.items():
        if not k in default_props.keys():
            other_props[k] = v
            del props[k]
    props['other'] = other_props

    nsubplots = props['subplot'][0] * props['subplot'][1]
    if nsubplots == len(arrays) and nsubplots > 100:
        print ('Warning: %d subplots seems like a few too many... '
               'I\'m going to assume you forgot that the first '
               'argument has to be a list or tuple.' % nsubplots)
        arrays = (arrays,)
        props['subplot'] = (1, 1)

    _initialize_subplots(len(arrays), props)
    narrays = len(arrays)
    npages = int(np.ceil(float(narrays) / nsubplots))
    
    # Pass by reference makes this tricky:
    # Cannot curry using lambda because it will only bind a reference
    # to x, not its value.  Using default arguments in the lambda
    # (lambda x=x:) fixes this because the default argument is
    # evaluated when the function is defined.  However this is much
    # uglier than functools.partial.
    plot_funcs = [functools.partial(_plotall_plot_page, arrays, props, x)
                  for x in xrange(npages)]
    if npages > 1:
        h = plot_pages(plot_funcs, not props['pub'])
    else:
        h = plot_funcs[0]()
    return h
        

def _initialize_subplots(narrays, props):
    nsubplots = props['subplot'][0] * props['subplot'][1]
    subplots = []
    for n in xrange(nsubplots):
        subplots.append((props['subplot'][0], props['subplot'][1], n + 1))

    plot_order = np.arange(nsubplots)
    if props['order'] == 'c':
        plot_order = plot_order.reshape(props['subplot']).flatten('F')

    # Only nsubplot plots can be shown at once.
    plot_num = plot_order[np.mod(np.arange(narrays), nsubplots)]
    props['subplots'] = np.asarray(subplots)[plot_num]

def _plotall_plot_page(arrays, props, curr_page=0):
    narrays = len(arrays)
    nsubplots = min(narrays, np.prod(props['subplot']))
    subplots = curr_page * nsubplots + np.arange(nsubplots)

    # Pass 1: plot everything and align axes.
    if props['clf']:
        plt.clf()
    all_axes = []
    all_image_axes = []
    click_handlers = {}
    for n, x in enumerate(subplots):
        if x < 0 or x >= narrays or arrays[x] is None:
            all_axes.append(None)
            continue

        kwargs = {}
        if all_axes:
            if 'x' in props['align']:
                kwargs['sharex'] = all_axes[0]
            if 'y' in props['align']:
                kwargs['sharey'] = all_axes[0]
        curr_axes = plt.subplot(*props['subplots'][x], **kwargs)

        data = np.asarray(arrays[x])
        if _plotall_get_prop(props['transpose'], x):
            data = np.transpose(data)

        plotfun = _plotall_get_prop(props['plotfun'], x)
        plotfun(data)

        clickfun = props['clickfun']
        if clickfun:
            click_handlers[curr_axes] = _plotall_get_prop(clickfun, x)

        all_axes.append(curr_axes)
        if data.ndim == 2:
            xlim_max = data.shape[1] - 0.5
        else:
            xlim_max = len(data)
        plt.setp(curr_axes, 'xlim', [-0.5, xlim_max])
        if data.ndim == 2:
            all_image_axes.append(curr_axes)
            plt.setp(curr_axes, 'ylim', [-0.5, data.shape[0]-0.5])

        # Draw colorbars on all subplots (even if they are not images)
        # to keep axis widths consistent.
        if props['colorbar']:
            plt.colorbar()

    if 'x' in props['align']:
        align_axes('x', all_axes)
    for ax in ('y', 'c'):
        if ax in props['align']:
            align_axes(ax, all_image_axes)

    fig = plt.gcf()
    clickfun = functools.partial(_plotall_click_handler,
                                 click_handlers=click_handlers)
    cid = fig.canvas.mpl_connect("button_press_event", clickfun)

    # Pass 2: set specified axis properties.
    for x in subplots:
        if x < 0 or x >= narrays or arrays[x] is None:
            continue
        curr_axes = all_axes[x]
        if curr_axes is None:
            continue
        plt.axes(curr_axes)

        succeeded = False
        for name, val in props['other'].iteritems():
            val = _plotall_get_prop(val, x)
            if isinstance(val, types.FunctionType):
                val = val()

            try:
                plt.setp(curr_axes, name, val)
                succeeded = True
            except AttributeError:
                for curr_axes_child in plt.get(curr_axes, 'children'):
                    try:
                        plt.setp(curr_axes_child, name, val)
                        succeeded = True
                    except:
                        pass
            if not succeeded:
                print 'Unable to set "%s" property on %s' % (name, curr_axes)
  
        if props['pub']:
            if x <= nsubplots - props['subplot'][1]:
                plt.xlabel('  ')
                #set(curr_axes, 'XTickLabel', ' ')
  
        plt.grid(props['grid'])

    # add_pan_and_zoom_controls_to_figure(properties.figure, all_axes);
    return all_axes

def _plotall_get_prop(prop, x):
    if isinstance(prop, list):
        prop = prop[min(x, len(prop)-1)]
    return prop

def _plotall_click_handler(event, click_handlers={}):
    if event.inaxes in click_handlers:
        plt.axes(event.inaxes)
        click_handlers[event.inaxes]()

def align_axes(axis_name='xyzc', axes=None):
    """Make sure that the given axes are aligned along the given axis_name
    ('x', 'y', 'c', or any combination thereof (e.g. 'xy' which is the
    default)).  If no axis handles are specified, all axes in the current
    figure are used.
    """
    if axes is None:
        axes = plt.findobj(match=plt.Axes)

    for name in axis_name:
        prop = '%clim' % name
        all_lim = []
        all_axes = []
        for ax in axes:
            if ax is None:
                continue
            try:
                all_lim.append(plt.get(ax, prop))
                all_axes.append(ax)
            except AttributeError:
                for childax in plt.get(ax, 'children'):
                    try:
                        all_lim.append(plt.get(childax, prop))
                        all_axes.append(childax)
                    except:
                        pass
        if all_lim:
            all_lim = np.asarray(all_lim)
            aligned_lim = (all_lim[:,0].min(), all_lim[:,1].max())
            plt.setp(all_axes, prop, aligned_lim)

