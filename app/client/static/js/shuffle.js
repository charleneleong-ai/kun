/**
 * File: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js/shuffle.js
 * Project: /Users/chaleong/Google Drive/engr489-2019/kun/app/client
 * Created Date: Monday, September 16th 2019, 3:42:24 pm
 * Author: Charlene Leong
 * -----
 * Last Modified: Wed Oct 02 2019
 * Modified By: Charlene Leong
 * -----
 * Copyright (c) 2019 Victoria University of Wellington ECS
 * ------------------------------------
 * Javascript will save your soul!
 */


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
        this._activeFilters = [];
        this.addFilterButtons();
        this.addSorting();
        this.addSearchFilter();

        document.addEventListener('keyup', this.selectedImgs.bind(this), false);

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

    _handleSortChange(evt) {
        // Add and remove `active` class from buttons.
        const buttons = Array.from(evt.currentTarget.children);
        buttons.forEach((button) => {
            if (button.querySelector('input').value === evt.target.value) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });

        // Create the sort options to give to Shuffle.
        const { value } = evt.target;
        let options = {};

        function sortByDate(element) {
            return element.getAttribute('data-created');
        }

        function sortByTitle(element) {
            return element.getAttribute('data-title').toLowerCase();
        }

        if (value === 'date-created') {
            options = {
                reverse: true,
                by: sortByDate,
            };
        } else if (value === 'title') {
            options = {
                by: sortByTitle,
            };
        }
        this.shuffle.sort(options);
    }

    // Advanced filtering
    addSearchFilter() {
        const searchInput = document.querySelector('.shuffle-search');
        if (!searchInput) {
            return;
        }
        searchInput.addEventListener('keyup', this._handleSearchKeyup.bind(this));
    }

    /**
     * Filter the shuffle instance by by cluster label
     * @param {Event} evt Event object.
     */
    _handleSearchKeyup(evt) {
        const searchText = evt.target.value.toLowerCase();
        
        // var filter
        
       
        // this.shuffle.filter(searchText)

        // var shuffleItems = this.shuffle.items

        
        // for (i = 0; i < shuffleItems.length; i++) {
        //     console.log(shuffleItems[i].element.getAttribute('data-groups'))
            
        // }
        this.shuffle.filter((element, shuffle) => {
            // If there is a current filter applied, ignore elements that don't match it.
            // const groups = JSON.parse(element.getAttribute('data-groups'));
            // console.log(groups)
            // const isElementInCurrentGroup = groups.indexOf(shuffle.group) !== -1;
            //     // Only search elements in the current group
            // if (!isElementInCurrentGroup) {
            //     return false;
            // }

            
            if (shuffle.group !== Shuffle.ALL_ITEMS) {
                // Get the item's groups.
                const groups = JSON.parse(element.getAttribute('data-groups'));
                const isElementInCurrentGroup = groups.indexOf(shuffle.group) !== -1;
                // Only search elements in the current group
                if (!isElementInCurrentGroup) {
                    return false;
                }
            }
            
            
            const c_label = element.getAttribute('data-groups')
            console.log(typeof(c_label), c_label)

            if (c_label.indexOf(searchText) !== -1) {
                console.log(c_label.indexOf(searchText) !== -1)
            }
            
            return c_label.indexOf(searchText) !== -1;
    
    
            // if (c_label.indexOf(searchText) !== -1) {
            //     console.log(c_label.indexOf(searchText) !== -1)
            // }
            
            // return c_label.indexOf(searchText) !== -1;
        });
    }
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
//   showProgress()
// };


ShuffleGrd.prototype.selectedImgs = function(evt) {
    this.refreshShuffle(document.getElementById('img-grd'))
    var shuffleItems = this.shuffle.items
    var selectedItems = document.getElementsByClassName('selected')

    if (evt.keyCode === 13 && selectedItems.length != 0) { // if space pressed
        var selectedImgIdx = []
        for (i = 0; i < selectedItems.length; i++) {
            selectedImgIdx.push(selectedItems[i].getAttribute('img_idx'))
        }
        console.log(selectedImgIdx)

        var imgGrdIdx = []
        for (i = 0; i < shuffleItems.length; i++) {
            if (selectedImgIdx.includes(shuffleItems[i].element.getAttribute('img_idx'))) {
                imgGrdIdx.push(i)
            }
        }
        console.log(imgGrdIdx)

        var imgIdx = []
        for (i = 0; i < shuffleItems.length; i++) {
            imgIdx.push(shuffleItems[i].element.getAttribute('img_idx'))
        }
        // console.log(imgIdx)

        $('#img-grd figure.selected').toggleClass('fadeout')
        $('#img-grd').toggleClass('shade')
        $('#som-status').toggleClass('shade')

        sendSelectedImgIdx(selectedImgIdx, imgGrdIdx, imgIdx)
    }
}


function sendSelectedImgIdx(selectedImgIdx, imgGridIdx, imgIdx) {
    $.ajax({
            url: `/selected/${selectedImgIdx}/${imgGridIdx}/${imgIdx}`,
            method: 'POST'
        })
        .done((res) => {
            console.log('SEEN img_idx: ' + res.img.selected_img_idx + ' img_grd_idx: ' + res.img.img_grd_idx + ' ' + res.img.seen + ' ' + res.img.num_seen)
            getStatus(res.task.task_type, res.task.task_id, res.task.task_data)
        })
        .fail((err) => {
            console.log(err)
        });
}


document.addEventListener('DOMContentLoaded', () => {
    window.shuffle = new ShuffleGrd(document.getElementById('img-grd'));
});