//함수선언
function doSomething(add) {
    console.log(add);
    const result = add(2, 3);
    console.log(result);
}

function add(a, b) {
    const sum = a + b;
    return sum;
}

//함수호출
doSomething(add);
