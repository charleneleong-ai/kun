


$( document ).ready(() => {
  console.log('Sanity Check!');
});

// $('#upload').on('click', function() {
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



var Shuffle = window.Shuffle;

var myShuffle = new Shuffle(document.querySelector('.shuffle-grid'), {
  itemSelector: '.js-item',
  sizer: '.my-sizer-element',
  buffer: 1,
});

// let images = getImagesFromDir(path.join(__dirname, 'imgs'));
// var images = getImagesFromDir('./imgs')
// console.log(images)
// // dirPath: target image directory
// function getImagesFromDir(dirPath) {
 
//     // All images holder, defalut value is empty
//     let allImages = [];
 
//     // Iterator over the directory
//     let files = fs.readdirSync(dirPath);
 
//     // Iterator over the files and push jpg and png images to allImages array.
//     for (file of files) {
//         let fileLocation = path.join(dirPath, file);
//         var stat = fs.statSync(fileLocation);
//         if (stat && stat.isDirectory()) {
//             getImagesFromDir(fileLocation); // process sub directories
//         } else if (stat && stat.isFile() && ['.jpg', '.png'].indexOf(path.extname(fileLocation)) != -1) {
//             allImages.push('static/'+file); // push all .jpg and .png files to all images 
//         }
//     }
 
//     // return all images in array format
//     return allImages;
// }