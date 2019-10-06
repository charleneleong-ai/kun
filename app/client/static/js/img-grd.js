/**
 * Created Date: Monday, October 7th 2019, 1:25:57 pm
 * Author: Charlene Leong leongchar@myvuw.ac.nz
 * Last Modified: Mon Oct 07 2019
 */

function refreshImgGrd(NUM_IMGS, NUM_FILTERED, NUM_REFRESH) {
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
            $('#num-seen').html('Images <b>[ ' + NUM_IMGS + ' ]</b>')
            $('#num-filtered').html('Filtered <b>[ ' + NUM_FILTERED + ' ]</b>')
            $('#num-refresh').html('Refresh <b>[ ' + NUM_REFRESH + ' ]</b>')
            $('#img-grd-wrapper').fadeIn()
            hide($('#num-selected'))
            // if ('{{c_label}}' == '{{C_LABEL}}'){
            //     $('#btn-{{c_label}}').addClass('active')
            // }
        })
        .fail((err) => {
            console.log(err)
        });
}

// Adding click listener to the cluster btns
document.querySelector('.btn-group').addEventListener('click', function(evt){
    target = evt.target.getAttribute('class')
    if (target.includes('btn--primary') && !target.includes('active')){
        $('#img-grd-wrapper').toggleClass('shade')
        c_label = evt.target.id.substring(4)
        console.log(c_label)
        $('.active').removeClass('active')
        $('#btn-'+c_label).addClass('active')
        
        taskData = { 'task_data': {'C_LABEL': c_label, 'SOM_MODE': 'switch'}}
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

