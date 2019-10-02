/**
 * File: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js/selection.js
 * Project: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js
 * Created Date: Monday, September 23rd 2019, 5:10:29 pm
 * Author: Charlene Leong
 * -----
 * Last Modified: Wed Oct 02 2019
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
    selectables: ['.shuffle-grd > figure'],

    // The container is also the boundary in this case
    boundaries: ['.shuffle-grd']
}).on('start', ({ inst, selected, oe }) => {
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


}).on('move', ({ changed: { removed, added } }) => {
    // Add a custom class to the elements that where selected.
    for (const el of added) {
        el.classList.add('selected');
    }

    // Remove the class from elements that where removed
    // since the last selection
    for (const el of removed) {
        el.classList.remove('selected');
    }

}).on('stop', ({ inst }) => {
    inst.keepSelection();

   
    showProgress()
});

// Remove selected when clicking outside img-grd
window.addEventListener('click', function(e) {
    if (document.getElementsByClassName('grd-item').length != 0 &&
        !document.getElementById('img-grd').contains(e.target)) {
        $('#img-grd figure.selected').toggleClass('selected')
        showProgress()
    }
});


function showProgress() {
    // Show Remove if at least one item is selected, else hide
    numSelected = document.getElementsByClassName('selected').length 
    if (numSelected == 0 || document.getElementsByClassName('grd-item').length == 0) {
        hide($('#progress'))
        hide($('#num-selected'))
        document.getElementById('num-selected').innerHTML = ''
    } else {
        show($('#progress'))
        show($('#num-selected'))
        document.getElementById('num-selected').innerHTML = numSelected
    }
}

function hide(elem){
    if (elem.attr('class').includes('show')){
        elem.removeClass('show')
        elem.addClass('hide')
    }
}


function show(elem){
    if (elem.attr('class').includes('hide')){
        elem.removeClass('hide')
        elem.addClass('show')
    }
}