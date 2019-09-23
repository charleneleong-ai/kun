/**
 * File: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js/selection.js
 * Project: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js
 * Created Date: Monday, September 23rd 2019, 5:10:29 pm
 * Author: Charlene Leong
 * -----
 * Last Modified: Mon Sep 23 2019
 * Modified By: Charlene Leong
 * -----
 * Copyright (c) 2019 Victoria University of Wellington ECS
 * ------------------------------------
 * Javascript will save your soul!
 */

console.log('SelectionJS')

const selection = Selection.create({

    // Class for the selection-area
    class: 'selection',

    // All elements in this container can be selected
    selectables: ['.shuffle-grid > figure'],

    // The container is also the boundary in this case
    boundaries: ['.shuffle-grid']
}).on('start', ({inst, selected, oe}) => {

    // Remove class if the user isn't pressing the control key or ⌘ key
    if (!oe.ctrlKey && !oe.metaKey) {

        // Unselect all elements
        for (const el of selected) {
            el.classList.remove('selected');
            inst.removeFromSelection(el);
        }

        // Clear previous selection
        inst.clearSelection();
    }

}).on('move', ({changed: {removed, added}}) => {

    // Add a custom class to the elements that where selected.
    for (const el of added) {
        el.classList.add('selected');
    }

    // Remove the class from elements that where removed
    // since the last selection
    for (const el of removed) {
        el.classList.remove('selected');
    }

}).on('stop', ({inst}) => {
    inst.keepSelection();
    showRemoveBtn() 
});

