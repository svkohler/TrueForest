var uploaded_drone_img; 

const DD_Area = document.querySelector('#drop_zone')


function dropHandler(ev) {
    console.log('File(s) dropped');

    // Prevent default behavior (Prevent file from being opened)
    ev.stopPropagation();
    ev.preventDefault();

    const fileList = ev.dataTransfer.files;

    readImage(fileList[0]);

    fileUpload.files = ev.dataTransfer.files;

    console.log(uploaded_drone_img)
    console.log('uploaded image.')
}

function dragOverHandler(ev) {
    console.log('File(s) in drop zone');
  
    // Prevent default behavior (Prevent file from being opened)
    ev.preventDefault();
  }


function readImage(file) {
    const reader = new FileReader();
    reader.addEventListener('load', (event) => {
    uploaded_drone_img = event.target.result;
    document.querySelector("#drop_zone").style.backgroundImage =`url(${uploaded_drone_img})`;
    });
    reader.readAsDataURL(file);
 }

function clearDD(ev){
    uploaded_drone_img = null
    document.querySelector('#drop_zone').style.backgroundImage = null
    document.getElementById('form_drone_img').value = null
}

function submitImgForm(){
    alert('test');
    document.getElementById("form_drone_img").submit();
}

function submitCoordinatesForm(){
    alert('test');
    document.getElementById("form_coordinates").submit();
}