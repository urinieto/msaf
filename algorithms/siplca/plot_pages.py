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

import tkSimpleDialog

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_pages(funcs, draw_pager_controls=True):
    """Sets up a figure with controls for browsing across multiple "pages"

    Each entry of the funcs array should be a function that takes no
    arguments and draws the contents of a particular page.

    E.g. plot_pages([lambda: plt.plot(x, y),
                     lambda: plt.imshow(np.random.rand(100,100))])

    will set up a figure containing plot(x,y) and controls that allow
    the user to switch to the next page.  When the "Next" button is clicked,
    the figure will be redrawn to contain a random 100x100 pixel image.

    Note that the funcs[i] is called every time the user switches to
    page i.  So in the previous example, the random image will change
    every time page 2 is redrawn (by interacting with the paging
    controls, not by underlying GUI updates).
    """
    return _plot_page(funcs, 0, draw_pager_controls)

def _plot_page(funcs, curr_page, draw_pager_controls=True):
    isinteractive = plt.isinteractive()
    if isinteractive:
        plt.ioff()
    plt.clf()
    h = funcs[curr_page]()
    if draw_pager_controls and len(funcs) > 1:
        _add_pager_controls(funcs, curr_page)
    plt.draw()
    if isinteractive:
        plt.ion()
    return h

def _add_pager_controls(funcs, curr_page):
    npages = len(funcs)
    bpos = np.asarray([0.0125, 0.0125, 0.06, 0.06]) 
    pos = bpos
    _create_pager_button(pos, 'First',
                         lambda ev: _plot_page(funcs, 0),
                         curr_page != 0)
    pos += [bpos[2], 0, 0, 0] 
    _create_pager_button(pos, 'Prev',
                         lambda ev: _plot_page(funcs, curr_page - 1),
                         curr_page > 0)
    pos += [bpos[2], 0, 0.05, 0]
    def open_page_dialog():
        page = tkSimpleDialog.askinteger('Jump to page',
                                         'Jump to page (1 -- %d)' % npages)
        if not page or page < 0 or page > npages:
            page = curr_page + 1
        return page - 1
    _create_pager_button(pos, '%d / %d' % (curr_page+1, npages),
                         lambda ev: _plot_page(funcs, open_page_dialog()),
                         True)
    pos += [bpos[2], 0, -0.05, 0] 
    _create_pager_button(pos, 'Next',
                         lambda ev: _plot_page(funcs, curr_page + 1),
                         curr_page < npages - 1)
    pos += [pos[2], 0, 0, 0] 
    _create_pager_button(pos, 'Last',
                         lambda ev: _plot_page(funcs, npages - 1),
                         curr_page != npages - 1)

def _create_pager_button(pos, label, fun, enabled=True):
    disabled_color = '#999999'
    ax = plt.axes(pos)
    b = mpl.widgets.Button(ax, label)
    if enabled:
        b.on_clicked(fun)
    else:
        b.color = disabled_color
        b.hovercolor = b.color
 
