//State: pending -> fulfilled or rejected
//Producer vs Consumer

//1. Producer(시간을 정해서 함수 호출할 수 있는 기능)
const promise = new Promise((resolve, reject) => {
    //doing some heavy work(network, read files)
    console.log("doing something...");
    setTimeout(() => {
        resolve("kelly");
        reject(new Error("no network"));
    }, 2000);
});

//2.Consumers:then, catch, finally
promise
    .then((value) => {
        console.log(value);
    })
    .catch((error) => {
        console.log(error);
    })
    .finally(() => {
        //수행에 상관없이 마지막에 나오는
        console.log("finally");
    });
//3. Promise chaining
const fetchNumber = new Promise((resolve, reject) => {
    setTimeout(() => resolve(1), 1000);
});
fetchNumber
    .then((num) => num * 2)
    .then((num) => num * 3)
    .then((num) => {
        return new Promise((resolve, reject) => {
            setTimeout(() => resolve(num - 1), 1000);
        });
    })
    .then((num) => console.log(num));
