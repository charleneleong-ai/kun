/**
 * File: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js/shuffle.js
 * Project: /Users/chaleong/Google Drive/engr489-2019/kun/app/client
 * Created Date: Monday, September 16th 2019, 3:42:24 pm
 * Author: Charlene Leong
 * -----
 * Last Modified: Fri Sep 20 2019
 * Modified By: Charlene Leong
 * -----
 * Copyright (c) 2019 Victoria University of Wellington ECS
 * ------------------------------------
 * Javascript will save your soul!
 */

$( document ).ready(() => {
  console.log('Shuffle');
});

class Demo {
  constructor(element) {
    this.element = element;
    this.shuffle = new Shuffle(element, {
      itemSelector: '.js-item',
      sizer: element.querySelector('.my-sizer-element'),
    });

    // Log events.
    this.addShuffleEventListeners();
    this._activeFilters = [];
    this.addFilterButtons();
    this.addSorting();
    this.addSearchFilter();
    //document.querySelector('#remove').addEventListener('click', this.onRemoveClick.bind(this));
    this.shuffle.element.addEventListener('click', this.onContainerClick.bind(this));

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

/**
 * Remove a shuffle item when it's clicked.
 * @param {Object} event Event object.
 */
Demo.prototype.onContainerClick = function (event) {
  // Bail in older browsers. https://caniuse.com/#feat=element-closest
  if (typeof event.target.closest !== 'function') {
    return;
  }
  var element = event.target.closest('.js-item');
  if (element !== null) {
    element_idx = element.getAttribute('img_idx')

    // Return all items with same idx
    var items = this.shuffle.items
    var indicesToRemove = [];
    for (i=0; i<items.length; i++){
      if(items[i].element.getAttribute('img_idx')==element_idx){
        indicesToRemove.push(i);
      }
    }
    // Make an array of elements to remove.
    var collection = indicesToRemove.map(function (index) {
      console.log()
      return this.shuffle.items[index].element;
    }, this);

    this.shuffle.remove(collection)

  }
}

// Demo.prototype.search = function (evt) {
//   var searchText = evt.target.value.toLowerCase();

//   this.shuffle.filter(function (element, shuffle) {
//     var titleElement = element.querySelector('.picture-item__title');
//     var titleText = titleElement.textContent.toLowerCase().trim();

//     return titleText.indexOf(searchText) !== -1;
//   });
// };

// // Randomly choose some elements to remove
// Demo.prototype.onRemoveClick = function () {
//   var total = this.shuffle.visibleItems;

//   // None left
//   if (!total) {
//     return;
//   }

//   var numberToRemove = Math.min(3, total);
//   var indiciesToRemove = [];

//   // This has the possibility to choose the same index for more than
//   // one in the array, meaning sometimes less than 3 will be removed
//   for (var i = 0; i < numberToRemove; i++) {
//     indiciesToRemove.push(this.getRandomInt(0, total - 1));
//   }

//   // Make an array of elements to remove.
//   var collection = indiciesToRemove.map(function (index) {
//     return this.shuffle.items[index].element;
//   }, this);

//   // Tell shuffle to remove them
//   this.shuffle.remove(collection);
// };

document.addEventListener('DOMContentLoaded', () => {
  window.demo = new Demo(document.getElementById('img-grid'));
});
