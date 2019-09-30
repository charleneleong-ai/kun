/**
 * File: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js/index.js
 * Project: /Users/chaleong/Google Drive/engr489-2019/kun/app/client
 * Created Date: Saturday, September 14th 2019, 4:24:23 pm
 * Author: Charlene Leong
 * -----
 * Last Modified: Mon Sep 30 2019
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



$('#upload').bind('click', function() {
    $.ajax({
            url: 'tasks/upload',
            method: 'POST'
        })
        .done((res) => {
            $('#progress').addClass('show')
            console.log(res)
            getStatus(res.data.task_type, res.data.task_id, res.data.task_data)

        })
        .fail((err) => {
            console.log(err)
        });
});

function getStatus(taskType, taskID, taskData) {
    $.ajax({
            url: `/tasks/${taskType}/${taskID}/${taskData}`,
            method: 'GET'
        })
        .done((res) => {
            //For task table
            document.getElementById('task_type').innerHTML = res.data.task_type;
            document.getElementById('task_id').innerHTML = res.data.task_id;
            document.getElementById('task_status').innerHTML = res.data.task_status;
            document.getElementById('task_data').innerHTML = res.data.task_data;
            document.getElementById('task_result').innerHTML = res.data.task_result;

            // For progress bar
            $('#img-grd figure.selected').toggleClass('selected')
            $('#progress').fadeIn()
            if (taskType === 'upload') {
                document.getElementsByClassName('grd-item').length = 0
                document.getElementById('progress').innerHTML = 'Uploading image bucket <b>[' + res.data.task_data + ']</b> ...'
            } else if (taskType === 'train') {
                document.getElementById('progress').innerHTML = 'Training autoencoder on <b>[' + res.data.task_data + ']</b> ...'
            } else if (taskType === 'cluster') {
                document.getElementById('progress').innerHTML = 'Clustering <b>['+res.data.task_data+']</b> with hdbscan ...'    
            } else if (taskType === 'som') {
                document.getElementById('progress').innerHTML = 'Loading image grid with self organising map ...'
                
            }

            const taskStatus = res.data.task_status;
            if (taskType === 'som' && taskStatus === 'finished') {
                // Reload page first time for grid to render
                if (document.getElementsByClassName('grd-item').length == 0) location.reload()

                refreshImgGrd(res.data.task_data)
                showProgress()
            }
            if (taskStatus === 'finished' || taskStatus === 'failed') {
                document.getElementById('progress').innerHTML = 'Press <b>[ ENTER ]</b> to remove'
                return false;
            }
            setTimeout(function() {
                getStatus(res.data.task_type, res.data.task_id,  res.data.task_data);
            }, 1000);

        })
        .fail((err) => {
            console.log(err)
        });
}

function refreshImgGrd(num_seen) {
    $.ajax({
            url: `/`,
            method: 'GET'
        })
        .done(function() {
            console.log('Reloading image grid...')
                // console.log($('#img-grd').attr('class'))
            $('#num-status').fadeOut();
            $('#img-grd').fadeOut();
            $('#img-grd ').toggleClass('shade') // Need to reset toggle class before reloading
            $('#img-grd').load(location.href + ' #img-grd>*', ''); //Reload img-grd div
            document.getElementById('num-seen').innerHTML = num_seen
            $('#num-status').fadeIn();
            $('#num-status').toggleClass('shade')
            $('#img-grd').fadeIn();
            
        })
        .fail((err) => {
            console.log(err)
        });

}