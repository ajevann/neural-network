var math = require('mathjs');
var fs = require('fs');

function normalize(nomenator, denominator){
  var xMax, yMax;

  if(typeof denominator === 'object'){
    xMax = denominator.valueOf()[0]; 
    yMax = denominator.valueOf()[1]; //make more dynamic
  }else if(typeof denominator === 'number'){
    xMax = yMax = denominator; //make more dynamic
  }

  if(nomenator.size && denominator.valueOf){ //check its a matrix
    var nSize = nomenator.size();
    
    var x, y;
    var i, j;

    for(x = 0; x < nSize[0]; x++){
      for(y = 0; y < nSize[1]; y++){
        var value, norm;
        if(y === 0){
          value = nomenator.get([x, y]);
          norm = value / xMax;
        }
        if(y === 1){
          value = nomenator.get([x, y]);
          norm = value / yMax;
        }

        nomenator.set([x, y], norm);
      }
    }
  }

  return nomenator;
}

function Neural_Network(){
  var self = this;

  self.init = function(){
    self.inputSize = 2;
    self.outputSize = 1;
    self.hiddenSize = 3;
    
    self.W1 = math.multiply(math.ones(self.inputSize, self.hiddenSize), Math.random());
    // self.W1 = math.multiply(math.ones(self.inputSize, self.hiddenSize), 0.5);
    self.W2 = math.multiply(math.ones(self.hiddenSize, self.outputSize), Math.random());
    // self.W2 = math.multiply(math.ones(self.hiddenSize, self.outputSize), 0.5);
  };
  
  self.forward = function(X){
     self.z = math.multiply(X, self.W1);
     self.z2 = self.sigmoid(self.z);
     self.z3 = math.multiply(self.z2, self.W2);

     o = self.sigmoid(self.z3);

     return o;
  };

  self.sigmoid = function(s){ //fix this to handle math.matrixsp
    var a = math.exp(math.multiply(s, -1));
    var b = math.add(a, 1);
    var c = math.dotDivide(1, b);
    return c;
  };

  self.sigmoidPrime = function(s){ //fix this to handle math.matrix
    var a = math.multiply(s, -1);
    var b = math.add(1, s);
    var c = math.dotMultiply(s, b);
    return c;
  };

  self.backward = function(X, y, o){
    self.o_error = math.subtract(y, o);
    var sigpr = self.sigmoidPrime(o);
    self.o_delta = math.dotMultiply(self.o_error, sigpr);

    var W2Transpose = math.transpose(self.W2);
    self.z2_error = math.multiply(self.o_delta, W2Transpose);
    self.z2_delta = math.multiply(self.z2_error, self.sigmoidPrime(self.z2));

    var a = math.transpose(X);
    var b = math.multiply(a, self.z2_delta);
    self.W1 = math.add(self.W1, b);
    
    a = math.transpose(self.z2);
    b = math.multiply(a, self.o_delta);
    self.W2 = math.add(self.W2, b);
  };

  self.train = function(X, y){
    var o = self.forward(X);
    self.backward(X, y, o);
  };

  self.saveWeights = function(){
    writeToFile('w1.txt', self.W1);
    writeToFile('w2.txt', self.W2);
  };

  self.predict = function(){
    // console.log("Input (scaled): " + xPredicted);
    console.log("Output: " + self.forward(xPredicted));
  };

  self.init();
}

function writeToFile(filename, contents){
  fs.writeFile(filename, contents.toString(), (err) => {
    if (err) throw err;
    // console.log('The file has been saved!');
  });
}

function run(){
  var NN = new Neural_Network();

  var i = 0;
  for (i = 0; i < 1000; i++){
    // console.log('Input (scaled): \n', X.toString());
    // console.log('Actual Output: \n', y.toString());
    // console.log('Predicted Output: \n', NN.forward(X).toString());
    // console.log('Loss: \n', math.mean(math.square(y - NN.forward(X))));
    // console.log('\n');

    NN.train(X, y);
  }

  NN.saveWeights();
  NN.predict();
}

var X = math.matrix([[2,9], [1,5], [3,6]]);
var y = math.matrix([[92], [86], [89]]);
var xPredicted = math.matrix([[4,1]]);

var xMax = math.max(X, 0);
X = normalize(X, xMax);

var xPredictedMax = math.max(xPredicted);
xPredicted = normalize(xPredicted, xPredictedMax);

var yMax = 100;
y = normalize(y, yMax);

var i;
for(i = 0; i < 100; i++){
  run();
}