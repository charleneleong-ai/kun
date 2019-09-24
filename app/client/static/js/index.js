

/**
 * File: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js/index.js
 * Project: /Users/chaleong/Google Drive/engr489-2019/kun/app/client
 * Created Date: Saturday, September 14th 2019, 4:24:23 pm
 * Author: Charlene Leong
 * -----
 * Last Modified: Tue Sep 24 2019
 * Modified By: Charlene Leong
 * -----
 * Copyright (c) 2019 Victoria University of Wellington ECS
 * ------------------------------------
 * Javascript will save your soul!
 */


var socket;
$(document).ready(() => {
  console.log('Sanity Check!');
  // socket = io.connect('http://' + document.domain + ':' + location.port);
  // socket.on('connect', function (msg) {
  //   console.log('Client connected!', msg);

  // });

  // socket.on('after connect', function (msg) {
  //   console.log('After connect', msg);
  // });
});



$('#upload').bind('click', function () {
  $.ajax({
    url: 'tasks/cluster',
    method: 'POST'
  })
  .done((res) => {
    console.log(res)
    getStatus(res.data.task_type, res.data.task_id)

  })
  .fail((err) => {
    console.log(err)
  });
});

function getStatus(taskType, taskID) {
  $.ajax({
    url: `/tasks/${taskType}/${taskID}`,
    method: 'GET'
  })
    .done((res) => {
      //For task table
      document.getElementById('task_type').innerHTML = res.data.task_type;
      document.getElementById('task_id').innerHTML = res.data.task_id;
      document.getElementById('task_status').innerHTML = res.data.task_status;
      document.getElementById('task_result').innerHTML = res.data.task_result;

      // For progress bar
      if (taskType === 'upload'){
        document.getElementById('progress').innerHTML = 'Uploading images with label '+res.data.task_data+' ...'
      }else if (taskType === 'train'){
        document.getElementById('progress').innerHTML = 'Training autoencoder '+res.data.task_data+' ...'
      }else if (taskType === 'cluster'){
        document.getElementById('progress').innerHTML = 'Clustering with hdbscan ...'
      }else if (taskType === 'som'){
        document.getElementById('progress').innerHTML = 'Loading image grid with self organising map ...'
      }

      const taskStatus = res.data.task_status;

      if (taskType === 'som' && taskStatus === 'finished'){
        reloadShuffleGrid()
      }
      
      if (taskStatus === 'finished' || taskStatus === 'failed') {
        document.getElementById('progress').innerHTML = ''
        return false;
      }
      
      setTimeout(function () {
        getStatus(res.data.task_type, res.data.task_id);
      }, 1000);

    })
    .fail((err) => {
      console.log(err)
    });
}

function reloadShuffleGrid(){
  $.ajax({
    url: `/`,
    method: 'GET'
  })
  .done(function () {
    console.log('Reloading shuffle grid...')
    $('#img-grd').load(location.href+' #img-grd>*','');  //Reload img-grd
  })
  .fail((err) => {
    console.log(err)
  });

}