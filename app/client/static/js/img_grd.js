/**
 * Created Date: Monday, October 7th 2019, 1:25:57 pm
 * Author: Charlene Leong leongchar@myvuw.ac.nz
 * Last Modified: Fri Oct 18 2019
 */

function refreshImgGrd(NUM_IMGS, NUM_FILTERED, NUM_REFRESH) {
    $.ajax({
            url: `/`,
            method: 'GET'
        })
        .done(function() {
            console.log('Reloading image grid ...')
            $('#img-grd-wrapper').fadeOut();
            $('#img-grd-wrapper').removeClass('shade') // Need to reset toggle class before reloading

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
            
            /// For shuffle filter
            // $('#cluster-filter').load(location.href + ' #cluster-filter>*',
            //     function(responseTxt, statusTxt, xhr) {
            //         if (statusTxt == "success") {
            //             show($('#cluster-filter'))
            //             // ShuffleInstance.addFilterButtons()
            //             // console.log("Reloaded filter options");
                        
            //         }
            //         if (statusTxt == "error") {
            //             console.log("Error: " + xhr.status + ": " + xhr.statusText);
            //         }
            //     })
            $('#num-imgs').html('Images <b>[ ' + NUM_IMGS + ' ]</b>')
            $('#num-imgs').val(NUM_IMGS)
            $('#num-filtered').html('Filtered <b>[ ' + NUM_FILTERED + ' ]</b>')
            $('#num-filtered').val(NUM_FILTERED)
            $('#num-refresh').html('Refresh <b>[ ' + NUM_REFRESH + ' ]</b>')
            $('#img-grd-wrapper').fadeIn()
            hide($('#num-selected'))

        })
        .fail((err) => {
            console.log(err)
        });
}

// Adding click listener to the cluster btns
document.querySelector('#cluster-filter .btn-group').addEventListener('click', function(evt){
    target = evt.target.getAttribute('class')
    if (target.includes('btn--primary') && !target.includes('active')){
        $('#img-grd-wrapper').addClass('shade')
        
        c_label = evt.target.id.substring(4)
        console.log(c_label)
        $('#cluster-filter .active').removeClass('active')
        $('#btn-'+c_label).addClass('active')
        $('#progress').html('Switching to cluster <b>['+c_label+' ]<b/>')
        taskData = { 'task_data': {'C_LABEL': c_label, 'SOM_MODE': 'switch', 'FILTERED': false}}
        $.ajax({
            url: `/tasks/som`,
            method: 'POST',
            contentType: 'application/json; charset=UTF-8',
            data: JSON.stringify(taskData),
            dataType: 'json',
            success: console.log(JSON.stringify(taskData))
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
    
});

document.addEventListener('keydown', function(evt) {
    var imgs = $('.grd-item')
    var selectedImgs = $('.selected')

    if (!($('#img-grd').attr('class').includes('shade'))){  //SOM not reloading

        var imgIdx = []
        for (i = 0; i < imgs.length; i++) {
            imgIdx.push(imgs[i].getAttribute('img_idx'))
        }

        if (validImgs(imgIdx) 
            && selectedImgs.length == 0                  // if selected items empty
            && evt.keyCode == 13 && evt.shiftKey){   // if SHIFT+ENTER pressed
                $('#img-grd figure.selected').fadeTo(0, 0.2)
                $('#img-grd-wrapper').addClass('shade')
                console.log(imgIdx)
                updateSOM(imgIdx, ',', ',')
            }

        if (validImgs(imgIdx) 
            && selectedImgs.length != 0    // if selected items not empty
            && evt.keyCode === 13 ){     // if ENTER pressed
            var selectedImgIdx = []
            for (i = 0; i < selectedImgs.length; i++) {
                selectedImgIdx.push(selectedImgs[i].getAttribute('img_idx'))
            }

            // If we want to grow the selected idx neighbourhood
            // var imgGrdIdx = []
            // for (i = 0; i < imgs.length; i++) {
            //     if (selectedImgIdx.includes(imgs[i].element.getAttribute('img_idx'))) {
            //         imgGrdIdx.push(i)
            //     }
            // }
            // console.log(imgGrdIdx)
            console.log(imgIdx)
            console.log(selectedImgIdx)
            
            $('#img-grd figure.selected').fadeTo(0, 0.2)
            $('#img-grd-wrapper').addClass('shade')
            updateSOM(imgIdx, selectedImgIdx, ',')
        }
    }
});


function updateSOM(imgIdx, selectedImgIdx, imgGridIdx) {
    taskData = {'task_data': {'SOM_MODE': 'update', 'FILTERED': false}}
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

function filteredSOM(btn_id){
    if (btn_id.includes('filtered')){
        if ($('#num-filtered').val() =='' ){
            $('#num-filtered').val($('#num-filtered').html().match(/\d+/)[0])
        }
        
        if ($('#num-filtered').val() ==0 ){
            console.log('There are no filtered images')
            $('#progress').html('There are no filtered images')
            show( $('#progress'))
            return
        }
        console.log(btn_id)
        $('#num-imgs').removeClass('active')
        $('#num-filtered').addClass('active')
        FILTERED = true
    }else if(btn_id.includes('imgs')){
        console.log(btn_id)
        $('#num-filtered').removeClass('active')
        $('#num-imgs').addClass('active')
        FILTERED = false
    }

    C_LABEL = $('#cluster-filter .active').val()

    taskData = {'task_data': {'C_LABEL': C_LABEL, 'SOM_MODE': 'switch', 'FILTERED': FILTERED}}
    $.ajax({
            url: `/tasks/som`,
            method: 'POST',
            contentType: 'application/json; charset=UTF-8',
            data: JSON.stringify(taskData),
            dataType: 'json',
            success: console.log(JSON.stringify(taskData))
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

function validImgs(imgIdx){
    if ($('#num-imgs').val() =='' ){
        $('#num-imgs').val($('#num-imgs').html().match(/\d+/)[0])
    }
    if ($('#num-imgs').val() - [...new Set(imgIdx)].length <=0 ){
        console.log('There are no more images to process')
        $('#progress').html('There are no more images to process')
        show( $('#progress'))
        return false
    }
    return true
}