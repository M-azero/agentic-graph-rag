/// <reference types="vite/client" />

// The product name, inlined by vite.config.ts from APP_NAME in the
// deployment's .env (default "Graph RAG").
declare const __APP_NAME__: string;

// The landing page's contact address (empty hides the link) and source link,
// from CONTACT_EMAIL / SOURCE_URL in .env.
declare const __CONTACT_EMAIL__: string;
declare const __SOURCE_URL__: string;
