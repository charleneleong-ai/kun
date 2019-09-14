


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