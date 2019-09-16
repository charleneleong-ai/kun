

/**
 * File: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js/index.js
 * Project: /Users/chaleong/Google Drive/engr489-2019/kun/app/client
 * Created Date: Saturday, September 14th 2019, 4:24:23 pm
 * Author: Charlene Leong
 * -----
 * Last Modified: Tue Sep 17 2019
 * Modified By: Charlene Leong
 * -----
 * Copyright (c) 2019 Victoria University of Wellington ECS
 * ------------------------------------
 * Javascript will save your soul!
 */



$( document ).ready(() => {
  console.log('Sanity Check!');
});


$('#upload').bind('click', function() {
  $.ajax({
    url: '/tasks/process_imgs',
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
    document.getElementById("task_type").innerHTML = res.data.task_type;
    document.getElementById("task_id").innerHTML = res.data.task_id;
    document.getElementById("task_status").innerHTML = res.data.task_status;
    document.getElementById("task_result").innerHTML = res.data.task_result;

    const taskStatus = res.data.task_status;
    if (taskStatus === 'finished' || taskStatus === 'failed') return false;
    setTimeout(function() {
      getStatus(res.data.task_type, res.data.task_id);
    }, 1000);
    
  })
  .fail((err) => {
    console.log(err)
  });
}

