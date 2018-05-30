

let data_xs = [];
let data_ys= [];
let dataset = [];


function openDataset(callback){
        // replace it with your file path in local server
        var url = "data.xlsx";
        var myobj;
        var oReq = new XMLHttpRequest();
        oReq.open("GET", url, true);
        oReq.responseType = "arraybuffer";
    
        oReq.onload = function(e) {
            var arraybuffer = oReq.response;
    
            /* convert data to binary string */
            var data = new Uint8Array(arraybuffer);
    
            var arr = new Array();
            for (var i = 0; i != data.length; ++i) {
                arr[i] = String.fromCharCode(data[i]);
            }
    
            var bstr = arr.join("");
    
            var cfb = XLSX.read(bstr, { type: 'binary' });
            var fieldsObjs = XLS.utils.sheet_to_json(cfb.Sheets["mlr04"]);

            myobj=  fieldsObjs;
            callback(myobj);
        }

        oReq.send();
 
}


//Weight and bias...
const w = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));



//Our predict function, with the input data as parameter 
//this return the Y what the model fit...
// Y = wx + b
function predict(x){
    return tf.tidy(()=>{
        return x.mul(w).add(b);
    });
}

//Calculate the error..Mean Squared Error (MSE)
function loss(predictions, labels){
    return tf.tidy(()=>{
        return predictions.sub(labels).square().mean();
    });
}

//Train function
function train(xs, ys,epochs=20){
    //Defining a optimizer..  Stochastic Gradient Descent(SGD)
    const learningRate = 0.001;
    const optimizer = tf.train.sgd(learningRate);
    for(let i = 0; i<epochs; i++){
        optimizer.minimize(()=>{
            const pys = predict(xs);
            return loss(pys, ys);
        });
    }
    console.log("after:");
    console.log("w:");
    w.print();
    console.log("b:");
    b.print();

    drawChart();
}

function sortFunction(myArray) { 
    return myArray.slice(0).sort(function(a, b){return a-b});;
  }


function  drawChart(){

    let newx = sortFunction(data_xs);
    let newy = [];
    newx.map((value)=>{
        dataset.map(function(register){
            if(register.X === value)
                newy.push(register.Y)
        });
    });

    let fxs = tf.tensor([0, 30]);
    let fys = predict(fxs);

    let mxs = [newx[0],newx[newx.length-1]];
    let myss = fys.dataSync();
    let mys = [];
    myss.map((value)=>{
        mys.push(Number(value));
    });

    var ctx = document.getElementById("myChart").getContext('2d');
    var myChart =   new Chart(document.getElementById("myChart"), {
    type: 'line',
    data: {
        labels: newx,
        datasets: [{ 
            data: newy,
            label: "Africa",
            borderColor: "#3e95cd",
            fill: false
        }
        ]
    },
    options: {
        title: {
        display: true,
        text: 'World population per region (in millions)'
        }
    }
    });

    var ctx = document.getElementById("myChart2").getContext('2d');
    var myChart =   new Chart(document.getElementById("myChart2"), {
    type: 'line',
    data: {
        labels: mxs,
        datasets: [{ 
            data: mys,
            label: "Africa",
            borderColor: "#3e95cd",
            tension:0,
            fill: false
        }
        ]
    },
    options: {
        title: {
        display: true,
        text: 'World population per region (in millions)'
        }
    }
    });


    fxs.dispose();
    fys.dispose();
}


var data = openDataset(function(data){
    dataset = data;
    // console.log(data);   
    //insert all the data in arrays
    data.map(function(register){
        data_xs.push(register.X);
        data_ys.push(register.Y);
    });
    //converting the arrays in tensors
    const xs = tf.tensor1d(data_xs);
    const ys = tf.tensor1d(data_ys);

    //print the info
    xs.print(); ys.print();
    console.log("Before:");
    console.log("w:");
    w.print();
    console.log("b:");
    b.print();

    //Train our model.
    train(xs,ys);

    xs.dispose();
    ys.dispose();

    
});
