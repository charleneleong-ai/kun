/**
 * Created Date: Friday, October 4th 2019, 2:28:44 pm
 * Author: Charlene Leong leongchar@myvuw.ac.nz
 * Last Modified: Mon Oct 07 2019
 */

// https://simonwep.github.io/selection/
 
$(document).ready(() => {
    console.log('SelectionJS')
});


const selection = Selection.create({
    // Class for the selection-area
    class: 'selection',
    // px, how many pixels the point should move before starting the selection
    startThreshold: 10,

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
    if (!($('#img-grd-wrapper').attr('class').includes('shade'))) {// SOM not reloading
        inst.keepSelection();
        // Fixing selection bug on shuffle filter
        // if ($('.active').length != 0){
        //     // console.log($('.active').attr('data-group'))
        //     // console.log($('.selected').length)
        //     activeGroup = $('.active').attr('data-group')
        //     $('.grd-item').each(function(idx){
        //         if (!this.getAttribute('data-groups').includes(activeGroup)){
        //             if (this.className.includes('selected')){
        //                 // console.log(this.getAttribute('data-groups'))
        //                 this.classList.remove('selected')
        //             }
        //         }
        //     })
        // }
        showRemove()
    }
});

// Remove selected when clicking outside img-grd
window.addEventListener('click', function(evt) {
    if (!($('#img-grd-wrapper').attr('class').includes('shade')) // SOM not reloading
        && $('.grd-item').length != 0   // Grid not empty
        && !$('#img-grd')[0].contains(evt.target)) {  // Clicked outside img-grd
        $('#img-grd figure.selected').toggleClass('selected')
        
        showRemove()
    }
});


function showRemove() {
    // Show Remove if at least one item is selected, else hide

    if (!($('#img-grd-wrapper').attr('class').includes('shade'))){
        numSelected = $('.selected').length 
        if (numSelected == 0 || $('.grd-item').length == 0) {
            $('#progress').html('Press <b>[ SHIFT + ENTER ]</b> to refresh all')
            show($('#progress'))
            // hide($('#progress'))
            hide($('#num-selected'))
            $('#num-selected').html('')
        } else {
            
            $('#progress').html('Press <b>[ ENTER ]</b> to remove selected')
            show($('#progress'))
            show($('#num-selected'))
            $('#num-selected').html(numSelected) 
        }
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