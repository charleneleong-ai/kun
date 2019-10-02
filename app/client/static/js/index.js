/**
 * File: /Users/chaleong/Google Drive/engr489-2019/kun/app/client/static/js/index.js
 * Project: /Users/chaleong/Google Drive/engr489-2019/kun/app/client
 * Created Date: Saturday, September 14th 2019, 4:24:23 pm
 * Author: Charlene Leong
 * -----
 * Last Modified: Wed Oct 02 2019
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
            method: 'POST',
        })
        .done((res) => {
            show($('#progress'))
            console.log(res)
            getStatus(res.task.task_type, res.task.task_id, res.task.task_data)
        })
        .fail((err) => {
            console.log(err)
        });
});

$('#train').bind('click', function() {
    $.ajax({
            url: 'tasks/train',
            method: 'POST',
        })
        .done((res) => {
            show($('#progress'))
            console.log(res)
            getStatus(res.task.task_type, res.task.task_id, res.task.task_data)
        })
        .fail((err) => {
            console.log(err)
        });
});

$('#cluster').bind('click', function() {
    $.ajax({
            url: 'tasks/cluster',
            method: 'POST'
        })
        .done((res) => {
            show($('#progress'))
            
            getStatus(res.task.task_type, res.task.task_id, res.task.task_data)
        })
        .fail((err) => {
            console.log(err)
        });
});

$('#som').bind('click', function() {
    //Resetting SOM for window refresh
    $.ajax({
            url: 'tasks/som',
            method: 'POST'
        })
        .done((res) => {
            show($('#progress'))
            console.log(res)
            getStatus(res.task.task_type, res.task.task_id, res.task.task_data)
        })
        .fail((err) => {
            console.log(err)
        });
});

function getStatus(taskType, taskID, taskData) {
    var taskData = {'task_data': taskData}
    $.ajax({
            url: `/tasks/${taskType}/${taskID}`,
            method: 'POST',
            contentType: 'application/json; charset=UTF-8',
            data: JSON.stringify(taskData),
            dataType: 'json',
            success: console.log(JSON.stringify(taskData))
        })
        .done((res) => {
            //For task table
            document.getElementById('task_type').innerHTML = res.task.task_type;
            document.getElementById('task_id').innerHTML = res.task.task_id;
            document.getElementById('task_status').innerHTML = res.task.task_status;
            document.getElementById('task_data').innerHTML = JSON.stringify(res.task.task_data);
            document.getElementById('task_result').innerHTML = res.task.task_result;

            // For progress bar
            $('#img-grd figure.selected').toggleClass('selected')
            show($('#progress'))
            if (taskType === 'upload') {
                document.getElementById('progress').innerHTML = 
                    'Uploading image bucket <b>[ '+res.task.task_data.label +' ]</b> ...'
            } else if (taskType === 'train') {
                document.getElementById('progress').innerHTML = 
                    'Training autoencoder on <b>[ '+res.task.task_data.label+' ]</b> with <b>[ '
                    +res.task.task_data.num_imgs+' ]</b> images ...'
            } else if (taskType === 'cluster') {
                document.getElementById('progress').innerHTML = 
                    'Clustering <b>[ '+res.task.task_data.label+' ]</b> with hdbscan ...'    
            } else if (taskType === 'som') {
                document.getElementById('progress').innerHTML = 
                    'Loading image grid with self organising map ...'
            }

            const taskStatus = res.task.task_status;
            if (taskType === 'som' && taskStatus === 'finished') {
                // Reload page if SOM is reset
                
                if (res.task.task_data.num_refresh==0) location.reload()
                refreshImgGrd(res.task.task_data.num_seen, res.task.task_data.num_filtered, res.task.task_data.num_refresh)
                showProgress()
                
            }
            if (taskStatus === 'finished' || taskStatus === 'failed') {
                document.getElementById('progress').innerHTML = 'Press <b>[ ENTER ]</b> to remove'
                return false;
            }
            setTimeout(function() {
                getStatus(res.task.task_type, res.task.task_id,  res.task.task_data);
            }, 1000);

        })
        .fail((err) => {
            console.log(err)
        });
}

function refreshImgGrd(num_seen, num_filtered, num_refresh) {
    $.ajax({
            url: `/`,
            method: 'GET'
        })
        .done(function() {
            console.log('Reloading image grid...')
            
            $('#img-grd').fadeOut();
            $('#img-grd ').toggleClass('shade') // Need to reset toggle class before reloading
            $('#som-status').toggleClass('shade')
            $('#img-grd').load(location.href + ' #img-grd>*', ''); //Reload img-grd div
            
            document.getElementById('num-seen').innerHTML = 'Images: ' + num_seen
            document.getElementById('num-filtered').innerHTML = 'Filtered: ' + num_filtered
            document.getElementById('num-refresh').innerHTML = 'Refresh: ' + num_refresh

            $('#img-grd').fadeIn();
            
        })
        .fail((err) => {
            console.log(err)
        });

}