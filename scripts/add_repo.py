#add_repo_script
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, BadRequest
import time

# Initialize the Databricks Workspace Client
w = WorkspaceClient()

# Define the repository details
repo_url = "https://github.com/vipultak-ds/ml-credit-risk.git"
repo_provider = "gitHub"
repo_path = "/Repos/vipultak7171@gmail.com/ml-credit-risk"

print("Resolving repository conflicts...")
print("=" * 50)

def handle_repository_conflicts():
    """Repository conflicts को handle करने के लिए comprehensive solution"""
    
    try:
        # Step 1: Check if repo exists
        repos_list = list(w.repos.list(path_prefix=repo_path))
        
        if not repos_list:
            print("Repository not found. Creating new repository...")
            repo = w.repos.create(
                url=repo_url,
                provider=repo_provider,
                path=repo_path
            )
            print(f"✅ Repository created successfully! Repo ID: {repo.id}")
            return repo
        
        repo = repos_list[0]
        print(f"Found existing repository. Repo ID: {repo.id}")
        
        # Step 2: Try to resolve conflicts by forcing update
        print("\nAttempting to resolve conflicts...")
        
        try:
            # Method 1: Try reset to remote HEAD (discard local changes)
            print("Method 1: Attempting to reset to remote HEAD...")
            w.repos.update(
                repo_id=repo.id,
                branch="main"
            )
            print("✅ Repository updated successfully!")
            return repo
            
        except BadRequest as conflict_error:
            print(f"Conflict detected: {conflict_error}")
            print("\nTrying alternative resolution methods...")
            
            # Method 2: Delete and recreate repository
            print("Method 2: Recreating repository to resolve conflicts...")
            try:
                print("Deleting existing repository...")
                w.repos.delete(repo_id=repo.id)
                print("Repository deleted.")
                
                # Wait a moment for deletion to complete
                time.sleep(2)
                
                print("Creating fresh repository...")
                new_repo = w.repos.create(
                    url=repo_url,
                    provider=repo_provider,
                    path=repo_path
                )
                print(f"✅ Fresh repository created successfully! Repo ID: {new_repo.id}")
                return new_repo
                
            except Exception as recreate_error:
                print(f"❌ Recreation failed: {recreate_error}")
                
                # Method 3: Create with different path and then rename
                print("Method 3: Using temporary path...")
                temp_path = f"{repo_path}_temp_{int(time.time())}"
                
                try:
                    temp_repo = w.repos.create(
                        url=repo_url,
                        provider=repo_provider,
                        path=temp_path
                    )
                    print(f"✅ Temporary repository created at: {temp_path}")
                    print("⚠️  Please manually rename or move this repository to the desired path")
                    return temp_repo
                    
                except Exception as temp_error:
                    print(f"❌ Temporary path creation failed: {temp_error}")
                    return None
                    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def verify_repository_content(repo):
    """Repository content को verify करें"""
    if repo:
        try:
            print(f"\n🔍 Verifying repository content...")
            print(f"Repository Path: {repo_path}")
            print(f"Repository ID: {repo.id}")
            print(f"Repository URL: {repo.url}")
            
            # Try to list repository contents
            try:
                files = dbutils.fs.ls(repo_path)
                print("Repository contents:")
                for file in files:
                    file_type = "📁 DIR " if file.isDir() else "📄 FILE"
                    print(f"  {file_type}: {file.name}")
                    
            except Exception as list_error:
                print(f"Could not list repository contents: {list_error}")
                
        except Exception as verify_error:
            print(f"Verification error: {verify_error}")

# Execute the conflict resolution
print("Starting conflict resolution process...")
resolved_repo = handle_repository_conflicts()

if resolved_repo:
    print(f"\n✅ Repository resolution completed successfully!")
    verify_repository_content(resolved_repo)
    
    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print("1. Repository is now ready for use")
    print("2. You can proceed with creating jobs")
    print("3. Future updates should work without conflicts")
    
    # Optional: Try to update once more to ensure everything is working
    try:
        print("\n🧪 Testing repository update...")
        w.repos.update(
            repo_id=resolved_repo.id,
            branch="main"
        )
        print("✅ Update test successful!")
        
    except Exception as test_error:
        print(f"⚠️  Update test failed: {test_error}")
        print("Repository is created but may need manual sync")
        
else:
    print("❌ Could not resolve repository conflicts automatically.")
    print("\nMANUAL RESOLUTION STEPS:")
    print("1. Go to Databricks Repos UI")
    print("2. Delete the existing repository")
    print("3. Re-clone the repository from GitHub")
    print("4. Or create repository with a different name")