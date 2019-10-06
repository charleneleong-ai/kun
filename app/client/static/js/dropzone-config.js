/**
 * Created Date: Saturday, October 5th 2019, 12:46:04 pm
 * Author: Charlene Leong leongchar@myvuw.ac.nz
 * Last Modified: Sun Oct 06 2019
 */


Dropzone.autoDiscover = false;

var DropzoneUpload = new Dropzone('#dropzone-upload', {
  maxFiles: 1,
  // parallelUploads: 100,
  // uploadMultiple:true,
  // maxFilesize: 1, //1 MB
  // acceptedFiles: '.png, .jpg, .jpeg',
  // autoProcessQueue: false,
  acceptedFiles: '.zip',
  thumbnailMethod: 'contain',
  init: function() {
    this.on("maxfilesexceeded", function(file){
        alert("You may only upload one zip at a time");
    });
  },
  chunksUploaded: (file, done) => {
    $('#progress-bar').css('width' , '100%');
    done()
  }
});


DropzoneUpload.on('queuecomplete', function(file) {
  $('#dropzone-upload').toggleClass('shade');
  $('#dropzone-upload').fadeOut();
  extract_zip()
  
});

DropzoneUpload.on('uploadprogress', (file, progress, bytesSent) => {
  if (file.upload.chunked &&  progress === 100 && 
    !!(file.upload.chunks.length <file.upload.totalChunkCount) 
    ) return
  $('#progress-bar').css('width' , progress + '%');
})


/// Functions for multi file img upload

// Excluding duplicate
// DropzoneUpload.on('addedfile', function(file) {
//   if (this.files.length) {
//       var _i, _len;
//       for (_i = 0, _len = this.files.length; _i < _len - 1; _i++) // -1 to exclude current file
//       {
//           if(this.files[_i].name === file.name && this.files[_i].size === file.size && this.files[_i].lastModifiedDate.toString() === file.lastModifiedDate.toString())
//           {
//               this.removeFile(file);
//           }
//       }
//   }
// });


// DropzoneUpload.on('totaluploadprogress', function(progress) {
//   $('#progress-bar').css('width' , progress + '%');
// });

// $('#upload-btn').click(function(){    
//   if ($('#dropzone-upload').get(0).dropzone.getAcceptedFiles().length == 1){
//     DropzoneUpload.processQueue();
//   }
// });
