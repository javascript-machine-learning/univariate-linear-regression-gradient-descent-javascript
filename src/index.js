import math from 'mathjs';
import csvToMatrix  from 'csv-to-array-matrix';

import {
  getDimensionSize,
  pushVector,
} from 'mathjs-util';

csvToMatrix('./src/data.csv', init, ',');

function init(matrix) {

  // Part 0: Preparation
  console.log('Part 0: Preparation ...\n');

  let X = math.eval('matrix[:, 1]', {
    matrix,
  });
  let y = math.eval('matrix[:, 2]', {
    matrix,
  });

  let m = getDimensionSize(y, 1);

  // Part 1: Cost
  console.log('Part 1: Cost ...\n');

  // Add Intercept Term
  X = pushVector(X, 0, math.ones([m, 1]).valueOf());

  let theta = [[-1], [2]];
  let J = computeCost(X, y, theta);

  console.log('Cost: ', J);
  console.log('with: ', theta);
  console.log('\n');

  theta = [[0], [0]];
  J = computeCost(X, y, theta);

  console.log('Cost: ', J);
  console.log('with: ', theta);
  console.log('\n');

  // Part 2: Gradient Descent
  console.log('Part 2: Gradient Descent ...\n');

  const ITERATIONS = 1500;
  const ALPHA = 0.01;

  theta = gradientDescent(X, y, theta, ALPHA, ITERATIONS);

  console.log('theta: ', theta);
}

function computeCost(X, y, theta) {
  let m = getDimensionSize(y, 1);

  let predictions = math.eval('X * theta', {
    X,
    theta,
  });

  let sqrErrors = math.eval('(predictions - y).^2', {
    predictions,
    y,
  });

  let J = math.eval(`1 / (2 * m) * sum(sqrErrors)`, {
    m,
    sqrErrors,
  });

  return J;
}

function gradientDescent(X, y, theta, ALPHA, ITERATIONS) {
  let m = getDimensionSize(y, 1);

  let thetaZero = theta[0];
  let thetaOne = theta[1];

  for (let i = 0; i < ITERATIONS; i++) {
    let predictions = math.eval('X * theta', {
      X,
      theta: [thetaZero, thetaOne],
    });

    thetaZero = math.eval(`thetaZero - ALPHA * (1 / m) * sum(predictions - y)`, {
      thetaZero,
      ALPHA,
      m,
      predictions,
      y,
    });

    thetaOne = math.eval(`thetaOne - ALPHA * (1 / m) * sum((predictions - y)' * X[:, 2])`, {
      thetaOne,
      ALPHA,
      m,
      predictions,
      y,
      X,
    });
  }

  return [thetaZero, thetaOne];
}

function gradientDescentAlternative(X, y, theta, ALPHA, ITERATIONS) {
  let m = getDimensionSize(y, 1);

  let thetaZero = theta[0];
  let thetaOne = theta[1];

  for (let i = 0; i < ITERATIONS; i++) {
    let predictions = math.eval('X * theta', {
      X,
      theta: [thetaZero, thetaOne],
    });

    thetaZero = math.eval(`thetaZero - ALPHA * (1 / m) * sum((predictions - y) .* X[:, 1])`, {
      thetaZero,
      ALPHA,
      m,
      predictions,
      y,
      X,
    });

    thetaOne = math.eval(`thetaOne - ALPHA * (1 / m) * sum((predictions - y) .* X[:, 2])`, {
      thetaOne,
      ALPHA,
      m,
      predictions,
      y,
      X,
    });
  }

  return [thetaZero, thetaOne];
}