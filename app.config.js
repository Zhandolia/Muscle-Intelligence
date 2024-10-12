import 'dotenv/config';

export default {
  expo: {
    name: "postureAI",
    slug: "posture-ai",
    extra: {
      clerkPublishableKey: process.env.EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY,
    },
  },
};
