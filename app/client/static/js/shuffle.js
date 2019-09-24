/**
 * File: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js/shuffle.js
 * Project: /Users/chaleong/Google Drive/engr489-2019/kun/app/client
 * Created Date: Monday, September 16th 2019, 3:42:24 pm
 * Author: Charlene Leong
 * -----
 * Last Modified: Tue Sep 24 2019
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

    document.addEventListener('keyup', this.removeImgs.bind(this), false);
 
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
    const searchInput = document.querySelector('.js-shuffle-search');
    if (!searchInput) {
      return;
    }
    searchInput.addEventListener('keyup', this._handleSearchKeyup.bind(this));
  }

  /**
   * Filter the shuffle instance by items with a title that matches the search input.
   * @param {Event} evt Event object.
   */
  _handleSearchKeyup(evt) {
    const searchText = evt.target.value.toLowerCase();
    this.shuffle.filter((element, shuffle) => {
      // If there is a current filter applied, ignore elements that don't match it.
      if (shuffle.group !== Shuffle.ALL_ITEMS) {
        // Get the item's groups.
        const groups = JSON.parse(element.getAttribute('data-groups'));
        const isElementInCurrentGroup = groups.indexOf(shuffle.group) !== -1;
        // Only search elements in the current group
        if (!isElementInCurrentGroup) {
          return false;
        }
      }
      const titleElement = element.querySelector('.picture-item__title');
      const titleText = titleElement.textContent.toLowerCase().trim();
      return titleText.indexOf(searchText) !== -1;
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

//   var removeIdx = []
//   for (i = 0; i < shuffleItems.length; i++) {
//     if (selectedImgIdx.includes(shuffleItems[i].element.getAttribute('img_idx'))) {
//       removeIdx.push(i)
//     }
//   }
//   console.log(removeIdx)
//   selectedImgs(selectedImgIdx, removeIdx)
//   showRemove()
// };


ShuffleGrd.prototype.removeImgs = function (evt) {
  if (evt.keyCode === 32){    // if space pressed
    this.refreshShuffle(document.getElementById('img-grd'))
    var shuffleItems = this.shuffle.items
    var selectedItems = document.getElementsByClassName('selected')
    var selectedImgIdx = []
    for (i = 0; i < selectedItems.length; i++) {
      selectedImgIdx.push(selectedItems[i].getAttribute('img_idx'))
    }
    console.log(selectedImgIdx)
  
    var imgIdx = []
    for (i = 0; i < shuffleItems.length; i++) {
      imgIdx.push(shuffleItems[i].element.getAttribute('img_idx'))
    }
    // console.log(imgIdx)
  
    var removeIdx = []
    for (i = 0; i < shuffleItems.length; i++) {
      if (selectedImgIdx.includes(shuffleItems[i].element.getAttribute('img_idx'))) {
        removeIdx.push(i)
      }
    }
    console.log(removeIdx)
    selectedImgs(selectedImgIdx, removeIdx)
    removeSelection()
  }

}


function selectedImgs(imgIdx, imgGridIdx) {
  $.ajax({
    url: `/selected/${imgIdx}/${imgGridIdx}`,
    method: 'POST'
  })
    .done((res) => {
      console.log('SEEN img_idx: ' + res.img.img_idx + ' img_grd_idx: ' + res.img.img_grd_idx + ' ' + res.img.seen + ' ' + res.img.num_seen)
      getStatus(res.data.task_type, res.data.task_id)
    })
    .fail((err) => {
      console.log(err)
    });
}


document.addEventListener('DOMContentLoaded', () => {
  window.shuffle = new ShuffleGrd(document.getElementById('img-grd'));
});

