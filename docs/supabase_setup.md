# Setting up Supabase for Long-Term Memory

The Home Generative Agent (HGA) can use Supabase as a persistent storage backend for long-term memory. This guide explains how to set up Supabase for use with HGA.

## Prerequisites

1. A Supabase account (sign up at https://supabase.com)
2. A Supabase project (create one from your Supabase dashboard)

## Setup Steps

1. Get your Supabase credentials:
   - Go to your Supabase project dashboard
   - Navigate to Project Settings > API
   - Copy the "Project URL" and "service_role key" (or anon/public key if you prefer)

2. Set up the database:
   - Go to the SQL Editor in your Supabase dashboard
   - Copy and paste the contents of `custom_components/home_generative_agent/supabase_setup.sql`
   - Click "Run" to execute the SQL commands

   This will:
   - Enable the pgvector extension for vector similarity search
   - Create the memories table with necessary columns
   - Create an index for efficient vector similarity search
   - Create a stored procedure for memory matching

3. Configure HGA:
   - In Home Assistant, go to Settings > Devices & Services
   - Find your Home Generative Agent integration
   - Click "Configure"
   - Uncheck "Use recommended settings"
   - Enter your Supabase Project URL and service role key
   - (Optional) Change the memory table name if you used a different name in step 2

## Table Structure

The memories table (`hga_memories`) has the following structure:

- `id`: UUID primary key
- `key`: Text identifier for the memory
- `value`: JSONB field containing the memory data
- `embedding`: Vector(512) for semantic search
- `created_at`: Timestamp of memory creation

## Security Considerations

- The service role key has full access to your database. Keep it secure and never expose it in client-side code.
- Consider using the anon/public key with Row Level Security (RLS) policies for production deployments.
- Regularly backup your Supabase database to prevent memory loss.

## Troubleshooting

1. If you see connection errors:
   - Verify your Project URL and key are correct
   - Check if your IP is allowed in Supabase's network restrictions

2. If vector similarity search fails:
   - Confirm the pgvector extension is enabled
   - Verify the `match_memories` function exists
   - Check the embedding dimensions match (should be 512)

3. For other issues:
   - Check the Home Assistant logs for detailed error messages
   - Verify your Supabase project's compute add-on has sufficient resources
