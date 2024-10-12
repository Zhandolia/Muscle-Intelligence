// import { useEffect } from 'react';
// import { useSegments, useRouter, useRootNavigationState } from 'expo-router'; // Or from your navigation library
// import { Session } from './types'; // Make sure you import the Session type or define it according to your auth system

// const useProtectedRoute = (session: Session | null) => {
//   const segments = useSegments(); // This provides the current navigation segments
//   const router = useRouter(); // This allows us to navigate
//   const navigationState = useRootNavigationState(); // Get the current root navigation state

//   useEffect(() => {
//     // Check if the navigation state has a valid key
//     if (!navigationState?.key) {
//       // If the key is not available, don't attempt navigation
//       return;
//     }

//     // Check if the user is in the auth group (e.g., login, signup pages)
//     const inAuthGroup = segments[0] === "(auth)";

//     // If the user is not authenticated and trying to access a protected route, redirect them to the auth page
//     if (!session && !inAuthGroup) {
//       router.replace("/auth");
//     }

//     // If the user is authenticated but is on the auth page (e.g., login/signup), redirect them to the home page
//     if (session && inAuthGroup) {
//       router.replace("/home"); // Adjust the route as necessary
//     }

//   }, [session, segments, navigationState?.key]); // Re-run the effect when session, segments, or navigation key changes
// };

// export default useProtectedRoute;
