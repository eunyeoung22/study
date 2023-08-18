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
    //ms:밀리 세컨즈
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

function popup() {
    window.open("https://www.google.com", "팝업", "width=500, height=300");
    window.scrollTo(100, 200);
}

function popClose() {
    window.close("popup.html");
}

const obj = { name: "철수" };
console.log("1:", obj);

const obj1 = {};
obj1.name = "철수";
console.log("2:", obj1.name);

const obj2 = { name: "영희" };
obj2.name = "철수";
console.log("3:", obj2.name);

const obj3 = {};
const obj4 = obj3;
obj4.name = "철수";
console.log("4:", obj4.name);
