//async & await
//clear style of using promise

//1. async
async function fetchUser() {
    //10초 걸리는 데이터
    return "kelly";
}

const user = fetchUser();
user.then(console.log);
console.log(user);

//2.await

function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
async function getApple() {
    await delay(1000);
    return "사과";
}

async function getBanana() {
    await delay(1000);
    return "바나나";
}

async function pickFruits() {
    const apple = await getApple();
    const banana = await getBanana();
    return `${apple} + ${banana}`;
}
pickFruits().then(console.log);
