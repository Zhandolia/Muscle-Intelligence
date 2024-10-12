// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// import { getFirestore } from "firebase/firestore";
// import { getStorage } from "firebase/storage";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDyXlsDqYznRTxKvpVR5cEl5X9ijDvJnYo",
  authDomain: "posture-ai.firebaseapp.com",
  projectId: "posture-ai",
  storageBucket: "posture-ai.appspot.com",
  messagingSenderId: "3395120463",
  appId: "1:3395120463:web:2b4309017895e8fe30693a",
};

const app = initializeApp(firebaseConfig);
export { app };
