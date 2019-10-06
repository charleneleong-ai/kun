/**
 * Created Date: Friday, October 4th 2019, 2:28:44 pm
 * Author: Charlene Leong leongchar@myvuw.ac.nz
 * Last Modified: Sun Oct 06 2019
 */


$(document).ready(() => {
    console.log('Sanity Check!!')
});


function extract_zip(){
    $('#instruct').html('')
    $.ajax({
            url: 'tasks/extract_zip',
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
    }

$('#train').bind('click', function() {
    $('#img-grd-wrapper').fadeOut()
    $('#instruct').html('')
    $('#progress').html('Training autoencoder ...')
    
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
    $('#img-grd-wrapper').toggleClass('shade')
    $('#instruct').html('')
    
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
    $('#img-grd-wrapper').toggleClass('shade')
    
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
    var taskData = { 'task_data': taskData }
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
            $('#task_type').html(res.task.task_type)
            $('#task_id').html(res.task.task_id)
            $('#task_status').html(res.task.task_status)
            $('#task_result').html(res.task.task_result)
            $('#task_data').html(JSON.stringify(res.task.task_data))

            updateProgress(res)

            task_status = res.task.task_status
            if (task_status === 'finished' || task_status === 'failed') {
                return false;
            }
            setTimeout(function() { // Poll every second
                getStatus(res.task.task_type, res.task.task_id, res.task.task_data);
            }, 1000);

        })
        .fail((err) => {
            console.log(err)
        });
}

function refreshImgGrd(NUM_SEEN, NUM_FILTERED, NUM_REFRESH) {
    $.ajax({
            url: `/`,
            method: 'GET'
        })
        .done(function() {
            console.log('Reloading image grid ...')
            $('#img-grd-wrapper').fadeOut();
            $('#img-grd-wrapper').toggleClass('shade') // Need to reset toggle class before reloading

            $('#img-grd').load(location.href + ' #img-grd>*',
                function(responseTxt, statusTxt, xhr) {
                    if (statusTxt == "success") {
                        ShuffleInstance.refreshShuffle($('#img-grd')[0])
                        console.log('Reloaded image grid')
                    }
                    if (statusTxt == "error") {
                        console.log("Error: " + xhr.status + ": " + xhr.statusText);
                    }
                });

            $('#cluster-filter').load(location.href + ' #cluster-filter>*',
                function(responseTxt, statusTxt, xhr) {
                    if (statusTxt == "success") {
                        show($('#cluster-filter'))
                        ShuffleInstance.addFilterButtons()
                        console.log("Reloaded filter options");
                    }
                    if (statusTxt == "error") {
                        console.log("Error: " + xhr.status + ": " + xhr.statusText);
                    }
                })

            $('#num-seen').html('Images <b>[ ' + NUM_SEEN + ' ]</b>')
            $('#num-filtered').html('Images <b>[ ' + NUM_FILTERED + ' ]</b>')
            $('#num-refresh').html('Refresh <b>[ ' + NUM_REFRESH + ' ]</b>')

            $('#img-grd-wrapper').fadeIn()

        })
        .fail((err) => {
            console.log(err)
        });
}

function updateProgress(res) {
    task_type = res.task.task_type
    LABEL = res.task.task_data.LABEL

    
    if (task_type === 'extract_zip') {
        progress_msg = res.task.task_data.progress_msg
        if (typeof(progress_msg) != 'undefined') $('#progress').html(progress_msg)
        
    } else if (task_type === 'load_data') {

        $('#label').html( LABEL)
        $('#progress').html('Loading image bucket ... <br/><b>[ ' + LABEL + ' ]</b>')

    } else if (task_type === 'train') {
        NUM_IMGS = res.task.task_data.NUM_IMGS
        NUM_TRAIN = res.task.task_data.NUM_TRAIN
        NUM_TEST = res.task.task_data.NUM_TEST

        progress_msg = 'Training autoencoder ... <br/> \
        <b>[ ' + NUM_IMGS + ' ]</b>  <b>[ ' + NUM_TRAIN + ' | ' + NUM_TEST + ' ]</b> images <br/> '

        if (typeof(test_loss = res.task.task_data.test_loss) != 'undefined') {
            BS = res.task.task_data.BS
            MAX_EPOCHS = res.task.task_data.MAX_EPOCHS
            LR = res.task.task_data.LR
            EPOCH = res.task.task_data.EPOCH
            epoch_progress = res.task.task_data.epoch_progress
            progress = res.task.task_data.progress
            train_loss = res.task.task_data.train_loss
            test_loss = res.task.task_data.test_loss
            PATIENCE = res.task.task_data.PATIENCE
            NUM_BAD_EPOCHS = res.task.task_data.NUM_BAD_EPOCHS

            progress_msg = progress_msg +
                'Batch size <b>[ ' + BS + ' ]</b> Learning rate <b>[ ' + LR + ' ]</b>  <br/>\
                 Epoch <b>[ ' + EPOCH + ' ]</b> ' + epoch_progress + ' ' + progress + '% <br/> \
                Train Loss <b>[ ' + train_loss + ' ]</b> Test Loss <b>[ ' + test_loss + ' ]</b><br/>'

            if (NUM_BAD_EPOCHS != 0) {
                progress_msg = progress_msg +
                    'Loss did not improve from ' + test_loss + ' for <b>[ ' + NUM_BAD_EPOCHS + ' ]</b> epochs'
            }
            $('#progress').html(progress_msg)

            percent = Math.round((EPOCH / MAX_EPOCHS) * 100)
            percent = percent + ((100 - percent) / PATIENCE) * NUM_BAD_EPOCHS // scale depending on num bad epochs
            $('#progress-bar').css('width', percent + '%')

            show($('.epoch-progress-bar-wrap'))
            animateProgressBar($('#epoch-progress-bar'), progress)
        }

    } else if (task_type === 'cluster') {
        $('#progress-bar').css('width', 100 + '%')
        hide($('.epoch-progress-bar-wrap'))

        progress_msg = 'Clustering with hdbscan ... <br/>'
        MIN_CLUSTER_SIZE = res.task.task_data.MIN_CLUSTER_SIZE
        if (typeof(MIN_CLUSTER_SIZE) != 'undefined') {
            progress_msg = progress_msg + 'Minimum cluster size <b>[ ' + MIN_CLUSTER_SIZE + ' ]</b> <br/>'
        }
        
        NUM_CLUSTERS = res.task.task_data.NUM_CLUSTERS
        if (typeof(NUM_CLUSTERS) != 'undefined') {
            progress_msg = progress_msg + '<b>[ ' + NUM_CLUSTERS + ' ]</b> clusters found <br/>'
        }

        _progress_msg = res.task.task_data.progress_msg
        if (typeof(_progress_msg) != 'undefined') {
            progress_msg = progress_msg + _progress_msg
        }

        $('#progress').html(progress_msg)

    } else if (task_type === 'som') {
        $('#img-grd figure.selected').toggleClass('selected')
        
        progress_msg = 'Loading image grid with self organising map ... <br/>'
        MAX_ITER = res.task.task_data.MAX_ITER
        DIMS = res.task.task_data.DIMS
        if (typeof(DIMS) != 'undefined') {
            progress_msg = progress_msg + '<b>[ ' + DIMS + ' ]</b>  <b>[ ' + MAX_ITER + ' ]</b> iterations <br/>'
        }

        NUM_ITER = res.task.task_data.NUM_ITER
        if (typeof(NUM_ITER) != 'undefined') {
            percent = Math.round((NUM_ITER / MAX_ITER) * 100)
            $('#progress-bar').css('width', percent + '%')
        }

        $('#progress').html(progress_msg)
    }

    task_status = res.task.task_status
    if (task_type === 'som' && task_status === 'finished') {
        $('#progress-bar').css('width', 100 + '%')
        hide($('#progress'))
        $('#progress').html('Press <b>[ ENTER ]</b> to remove')
        
        // Reload page if SOM is reset
        NUM_SEEN = res.task.task_data.NUM_SEEN
        NUM_FILTERED = res.task.task_data.NUM_FILTERED
        NUM_REFRESH = res.task.task_data.NUM_REFRESH

        if (NUM_REFRESH == 0) location.reload()
        refreshImgGrd(NUM_SEEN, NUM_FILTERED, NUM_REFRESH)
        showRemove()
    }
    show($('#progress'))
    
}

function animateProgressBar(progressBar, numIteration, index = 1) {
    width = (index / numIteration) * 100;
    progressBar.css('width', width + '%');
    if (index < numIteration) setTimeout(animateProgressBar, 50, progressBar, numIteration, index + 1);
}