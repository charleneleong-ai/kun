/**
 * Created Date: Friday, October 4th 2019, 2:28:44 pm
 * Author: Charlene Leong leongchar@myvuw.ac.nz
 * Last Modified: Mon Oct 07 2019
 */

// https://vestride.github.io/Shuffle/


$(document).ready(() => {
    console.log('ShuffleGrd');
});

class ShuffleGrd {
    constructor(element) {
        this.element = element;
        this.shuffle = new Shuffle(element, {
            itemSelector: '.grd-item',
            sizer: element.querySelector('.my-sizer-element'),
            speed: 300,
            staggerAmount: 30
        });

        // Log events.
        this.addShuffleEventListeners();
        // this._activeFilters = [];
        // this.addFilterButtons();
        // this.addSorting();
        // this.addSearchFilter();

        document.addEventListener('keyup', this.selectedImgs.bind(this), false);

    }

    refreshFilter(){
        this._activeFilters = [];
        this.addFilterButtons();
    }

    refreshShuffle(element) {
        this.shuffle = new Shuffle(element, {
            itemSelector: '.grd-item',
            sizer: element.querySelector('.my-sizer-element'),
            speed: 300,
            staggerAmount: 30
        });
    }

    /**
     * Shuffle uses the CustomEvent constructor to dispatch events. You can listen
     * for them like you normally would (with jQuery for example).
     */
    addShuffleEventListeners() {
        this.shuffle.on(Shuffle.EventType.LAYOUT, (data) => {
            console.log('layout. data:', data);
        });
        this.shuffle.on(Shuffle.EventType.REMOVED, (data) => {
            console.log('removed. data:', data);
        });
    }

    addFilterButtons() {
        const options = document.querySelector('.filter-options');
        if (!options) {
            return;
        }

        const filterButtons = Array.from(options.children);
        const onClick = this._handleFilterClick.bind(this);
        filterButtons.forEach((button) => {
            button.addEventListener('click', onClick, false);
        });
    }

    _handleFilterClick(evt) {
        const btn = evt.currentTarget;
        const isActive = btn.classList.contains('active');
        const btnGroup = btn.getAttribute('data-group');
        this._removeActiveClassFromChildren(btn.parentNode);

        let filterGroup;
        if (isActive) {
            btn.classList.remove('active');
            filterGroup = Shuffle.ALL_ITEMS;
        } else {
            btn.classList.add('active');
            filterGroup = btnGroup;
        }

        this.shuffle.filter(filterGroup);
    }

    _removeActiveClassFromChildren(parent) {
        const { children } = parent;
        for (let i = children.length - 1; i >= 0; i--) {
            children[i].classList.remove('active');
        }
    }

    addSorting() {
        const buttonGroup = document.querySelector('.sort-options');
        if (!buttonGroup) {
            return;
        }
        buttonGroup.addEventListener('change', this._handleSortChange.bind(this));
    }

    // _handleSortChange(evt) {
    //     // Add and remove `active` class from buttons.
    //     const buttons = Array.from(evt.currentTarget.children);
    //     buttons.forEach((button) => {
    //         if (button.querySelector('input').value === evt.target.value) {
    //             button.classList.add('active');
    //         } else {
    //             button.classList.remove('active');
    //         }
    //     });

    //     // Create the sort options to give to Shuffle.
    //     const { value } = evt.target;
    //     let options = {};

    //     function sortByDate(element) {
    //         return element.getAttribute('data-created');
    //     }

    //     function sortByTitle(element) {
    //         return element.getAttribute('data-title').toLowerCase();
    //     }

    //     if (value === 'date-created') {
    //         options = {
    //             reverse: true,
    //             by: sortByDate,
    //         };
    //     } else if (value === 'title') {
    //         options = {
    //             by: sortByTitle,
    //         };
    //     }
    //     this.shuffle.sort(options);
    // }

    // // Advanced filtering
    // addSearchFilter() {
    //     const searchInput = document.querySelector('.shuffle-search');
    //     if (!searchInput) {
    //         return;
    //     }
    //     searchInput.addEventListener('keyup', this._handleSearchKeyup.bind(this));
    // }

    // /**
    //  * Filter the shuffle instance by by cluster label
    //  * @param {Event} evt Event object.
    //  */
    // _handleSearchKeyup(evt) {
    //     const searchText = evt.target.value.toLowerCase();
    //     this.shuffle.filter((element, shuffle) => {
    //         // If there is a current filter applied, ignore elements that don't match it.
    //         // const groups = JSON.parse(element.getAttribute('data-groups'));
    //         // console.log(groups)
    //         // const isElementInCurrentGroup = groups.indexOf(shuffle.group) !== -1;
    //         //     // Only search elements in the current group
    //         // if (!isElementInCurrentGroup) {
    //         //     return false;
    //         // }

    //         if (shuffle.group !== Shuffle.ALL_ITEMS) {
    //             // Get the item's groups.
    //             const groups = JSON.parse(element.getAttribute('data-groups'));
    //             const isElementInCurrentGroup = groups.indexOf(shuffle.group) !== -1;
    //             // Only search elements in the current group
    //             if (!isElementInCurrentGroup) {
    //                 return false;
    //             }
    //         }
    //         const c_label = element.getAttribute('data-groups')
    //         console.log(typeof(c_label), c_label)

    //         if (c_label.indexOf(searchText) !== -1) {
    //             console.log(c_label.indexOf(searchText) !== -1)
    //         }
    //         return c_label.indexOf(searchText) !== -1;
    
    //     });
    // }
}

// ShuffleGrd.prototype.onRemoveClick = function () {

//   this.refreshShuffle(document.getElementById('img-grd'))
//   var shuffleItems = this.shuffle.items
//   var selectedItems = document.getElementsByClassName('selected')
//   var selectedImgIdx = [] 
//   for (i=0; i<selectedItems.length; i++){
//     selectedImgIdx.push(selectedItems[i].getAttribute('img_idx'))
//   }
//   console.log(selectedImgIdx)

//   var imgIdx = [] 
//   for (i = 0; i < shuffleItems.length; i++) {
//     imgIdx.push(shuffleItems[i].element.getAttribute('img_idx'))
//   }
//   // console.log(imgIdx)

//   var imgGrdIdx = []
//   for (i = 0; i < shuffleItems.length; i++) {
//     if (selectedImgIdx.includes(shuffleItems[i].element.getAttribute('img_idx'))) {
//       imgGrdIdx.push(i)
//     }
//   }
//   console.log(imgGrdIdx)
//   selectedImgs(selectedImgIdx, imgGrdIdx)
//   showRemove()
// };


ShuffleGrd.prototype.selectedImgs = function(evt) {
    var shuffleItems = this.shuffle.items
    var selectedItems = $('.selected')

    if (!($('#img-grd').attr('class').includes('shade'))){  //SOM not reloading

        var imgIdx = []
        for (i = 0; i < shuffleItems.length; i++) {
            imgIdx.push(shuffleItems[i].element.getAttribute('img_idx'))
        }

        if (selectedItems.length == 0                  // if selected items empty
            && evt.keyCode == 13 && evt.shiftKey){   // if SHIFT+ENTER pressed
                
                $('#img-grd figure.selected').fadeTo(0, 0.2)
                $('#img-grd-wrapper').toggleClass('shade')

                updateSOM(imgIdx, ',', ',')
            }

        if (selectedItems.length != 0    // if selected items not empty
            && evt.keyCode === 13 ){     // if ENTER pressed
            var selectedImgIdx = []
            for (i = 0; i < selectedItems.length; i++) {
                selectedImgIdx.push(selectedItems[i].getAttribute('img_idx'))
            }
            console.log(selectedImgIdx)

            // If we want to grow the selected idx neighbourhood
            // var imgGrdIdx = []
            // for (i = 0; i < shuffleItems.length; i++) {
            //     if (selectedImgIdx.includes(shuffleItems[i].element.getAttribute('img_idx'))) {
            //         imgGrdIdx.push(i)
            //     }
            // }
            // console.log(imgGrdIdx)

            $('#img-grd figure.selected').fadeTo(0, 0.2)
            $('#img-grd-wrapper').toggleClass('shade')

            updateSOM(imgIdx, selectedImgIdx, ',')

        }
    }
}


function updateSOM(imgIdx, selectedImgIdx, imgGridIdx) {
    taskData = { 'task_data': {'SOM_MODE': 'update'}}
    $.ajax({
            url: `/update_som/${imgIdx}/${selectedImgIdx}/${imgGridIdx}`,
            method: 'POST',
            contentType: 'application/json; charset=UTF-8',
            data: JSON.stringify(taskData),
            dataType: 'json',
            success: console.log(JSON.stringify(taskData))
        })
        .done((res) => {
            console.log('SEEN img_idx: ' + res.img.selected_img_idx + ' img_grd_idx: ' + res.img.img_grd_idx + ' ' + res.img.seen + ' ' + res.img.NUM_IMGS)
            getStatus(res.task.task_type, res.task.task_id, res.task.task_data)
        })
        .fail((err) => {
            console.log(err)
        });
}


var ShuffleInstance = window.shuffle;

document.addEventListener('DOMContentLoaded', () => {
    ShuffleInstance = new ShuffleGrd($('#img-grd')[0]);
});


